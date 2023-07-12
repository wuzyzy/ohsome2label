import json
import os
from datetime import datetime

import geojson

import shutil
from geojson import FeatureCollection, Feature
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon, box, shape
from shapely.strtree import STRtree
from tqdm import tqdm

from ohsome2label.palette import palette
from ohsome2label.tile import Bbox, apply_transform, get_bbox, tile_get_transform, xy
from ohsome2label.utils import get_area

nx = 256
ny = 256


class TaskError(Exception):
    """Wrong task"""

    pass


def bounds_to_bbox(bounds):
    (minx, miny, maxx, maxy) = bounds
    bbox = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    return bbox


def gen_anno(coords, idx, imgIdx, catIdx):
    anno = {}
    anno["id"] = idx
    anno["category_id"] = catIdx
    anno["iscrowd"] = 0
    seg = []
    anno["image_id"] = imgIdx
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    for _x, _y in zip(xs, ys):
        seg.append(_x)
        seg.append(_y)
    anno["segmentation"] = [seg]
    anno["area"] = get_area(xs, ys)
    anno["bbox"] = [max(xs), max(ys), min(xs), min(ys)]
    return anno


def expolde_multipolygon(multipoly):
    coords_list = []
    for poly in multipoly:
        coords = list(poly.exterior.coords)
        coords_list.append(coords)
    return coords_list


def parse_polygon(coordinates, trans, nx=256, ny=256):
    """parse polygon

    :param coordinates: wgs84 coordinates
    :param trans: translation to get image-based coordinate system coordinate
    :param nx: image width
    :param ny: image length
    """
    exterior = coordinates[0]
    exterior = [xy(*coord) for coord in exterior]
    exterior = apply_transform(exterior, trans)

    if len(coordinates) > 1:
        interiors = coordinates[1:]
        interiors = [[xy(*coord) for coord in interior] for interior in interiors]
        interiors = [apply_transform(interior, trans) for interior in interiors]
    else:
        interiors = []

    return Polygon(exterior, interiors).buffer(0)


def check_topo(feats, tile, nx=256, ny=256):
    """check features geometry, due to coco cannot recognize segmentation with
       holes, return sorted list of (geometry, label) tuple

    :param feats: geojson feature
    :param tile: tile of the feature
    :param nx: image width
    :param ny: image length
    """
    trans = tile_get_transform(tile, nx, ny)
    bbox = box(0, 0, 255, 255)

    # geoms element is (geometry, label) tuple
    geoms = []

    # construct tile-based coordinate-system shapely geometry
    idx = 0
    for feat in feats:
        idx += 1
        geometry = feat["geometry"]
        label = feat["properties"]["label"]
        if geometry["type"] == "Polygon":
            coordinates = geometry["coordinates"]
            poly = parse_polygon(coordinates, trans, nx, ny)
            geoms.append((poly.area, poly, label))
        elif feat["geometry"]["type"] == "MultiPolygon":
            coordinates = geometry["coordinates"]
            try:
                poly = MultiPolygon(
                    [parse_polygon(coords, trans, nx, ny) for coords in coordinates]
                )
                geoms.append((poly.area, poly, label))
            except Exception:
                assert 0

    # put larger area geometry on the top of list
    geoms.sort(key=lambda x: x[0], reverse=True)

    # Due to coco not support segmentation with holes.
    # So the exterior will be used as the segmentation
    for _, geom, label in geoms:
        geom = bbox.intersection(geom).buffer(0)
        if geom.area > 0:
            yield (geom, label)


def burn_tile(geoms, task, pal, fname, nx=256, ny=256):
    """Burn a tile

    :param geoms: (geom, label) tuple
    :param pal: palette
    :param fname: path to store the output image
    :param nx: image width
    :param ny: image length
    """
    draws = []
    im = Image.new(mode="RGB", size=(nx, ny), color="#000000")
    draw = ImageDraw.Draw(im)
    for geom, label in geoms:
        color = pal.color(label)
        if task == "segmentation":
            if geom.type == "Polygon":
                draws.append((list(geom.exterior.coords), color))
            elif geom.type == "MultiPolygon":
                draws += [(list(g.exterior.coords), color) for g in geom]
        elif task == "object detection":
            bbox = bounds_to_bbox(geom.bounds)
            draws.append((bbox, color))
        else:
            raise TaskError

    for idx, (coords, color) in enumerate(draws):
        if task == "segmentation":
            draw.polygon(coords, fill=color)
        elif task == "object detection":
            draw.line(coords)

        yield (idx, label, coords)

    im.save(fname, "PNG")


class geococo(object):
    """class for generate geo coco
    https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
    """

    def __init__(self, config):
        self.tags_to_cats(config.tags)

        # coco info
        info = {}
        info["description"] = config.name
        info["date_created"] = datetime.now().strftime("%Y/%m/%d")
        self.info = info

        # coco licenses
        self.lic = [
            {
                "url": "http://creativecommons.org/licenses/by/2.0/",
                "id": 4,
                "name": "Attribution License",
            }
        ]

        self.imgs = []
        self.annos = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True

    def tags_to_cats(self, _tags):
        """Convert tags to coco categories."""
        self.catIdxs = {}
        self.cats = []

        _label = {}

        tags = []
        for _t in _tags:
            if "__main." in _t["label"]:
                _t["label"] = _t["label"].replace("__main.", "")
                tags.append(_t)
            elif "__sub." in _t["label"]:
                continue
            else:
                tags.append(_t)
        for tag in tags:
            if tag["label"] not in _label:
                _label[tag["label"]] = len(_label)

        for tag in tags:
            cat = {}
            cat["supercategory"] = tag["label"]
            cat["id"] = _label[tag["label"]] + 1
            cat["name"] = tag["label"]
            if cat in self.cats:
                continue
            else:
                self.cats.append(cat)
                self.catIdxs[tag["label"]] = cat["id"]

    def to_json(self):
        coco = {}
        coco["info"] = self.info
        coco["licenses"] = self.lic
        coco["images"] = self.imgs
        coco["annotations"] = self.annos
        coco["categories"] = self.cats
        return coco


def gen_label(cfg, workspace):
    """Generate label and annotations in coco format

    :param cfg: ohsome2label config
    :param workspace: workspace
    """
    tile_dir = workspace.tile
    img_dir = workspace.label
    geoms = []
    feats = {}
    tile_feats = {}

    # open downloaded geojson file
    if cfg.api == "ohsome":
        for tag in cfg.tags:
            if "__main" in tag["label"]:
                main_geojson = "{lab}_{k}_{v}.geojson".format(
                    lab=tag["label"], k=tag["key"], v=tag["value"]
                )
                output_geojson = main_geojson.replace("__main.", "")
                sub_label = tag["label"].replace("__main.", "__sub.")
                sub_geojson = ""
                for _t in cfg.tags:
                    if _t["label"] == sub_label:
                        sub_geojson = "{lab}_{k}_{v}.geojson".format(
                            lab=_t["label"], k=_t["key"], v=_t["value"]
                        )
                if sub_geojson == "":
                    if not os.path.exists(os.path.join(workspace.raw, output_geojson)):
                        shutil.copy(
                            os.path.join(workspace.raw, main_geojson),
                            os.path.join(workspace.raw, output_geojson),
                        )
                else:
                    label = tag["label"].split(".")[-1]
                    if not os.path.exists(os.path.join(workspace.raw, output_geojson)):
                        print(f"Start to merge {label} geojson")
                        _overlap(
                            os.path.join(workspace.raw, main_geojson),
                            os.path.join(workspace.raw, sub_geojson),
                            os.path.join(workspace.raw, output_geojson),
                            label,
                        )
                fname = output_geojson
            elif "__sub" in tag["label"]:
                continue
            else:
                label = tag["label"]
                fname = "{lab}_{k}_{v}.geojson".format(
                    lab=tag["label"], k=tag["key"], v=tag["value"]
                )
            fpath = os.path.join(workspace.raw, fname)
            # get valid tile
            with open(fpath, encoding="utf-8") as f:
                data = geojson.loads(f.read().replace("'", ""))
                features = data["features"]
                for feature in features:
                    geom = shape(feature["geometry"])
                    geoms.append(geom)
                    feature["properties"]["label"] = label
                    feats[geom.to_wkb()] = feature
    elif cfg.api == "overpass":
        fname = "overpass_query.geojson"
        fpath = os.path.join(workspace.raw, fname)
        with open(fpath, encoding="utf-8") as f:
            data = geojson.loads(f.read().replace("'", ""))
            features = data["features"]
            for feature in features:
                for tag in cfg.tags:
                    key = tag.get("key", "")
                    value = tag.get("value", "")
                    if value == "" and key in feature["properties"]:
                        geom = shape(feature["geometry"])
                        geoms.append(geom)
                        feature["properties"]["label"] = tag["label"]
                        feats[geom.to_wkb()] = feature
                        break
                    elif feature["properties"].get(key, "") == value:
                        geom = shape(feature["geometry"])
                        geoms.append(geom)
                        feature["properties"]["label"] = tag["label"]
                        feats[geom.to_wkb()] = feature
                        break

    print("Start to build geom tree")
    tree = STRtree(geoms)

    # clip by tile into small tile geojson

    tile_dir = workspace.tile
    list_path = os.path.join(workspace.other, "tile_list")

    with open(list_path, "w", encoding="utf-8") as f:
        for _t in tqdm(cfg.tiles):
            _box = box(*get_bbox(_t))
            r = tree.query(_box)

            if len(r) != 0:
                tile = tile_feats.get(_t, [])
                for g in r:
                    tile.append(feats[g.to_wkb()])
                    tile_feats[_t] = tile
                tile_name = "{0.z}.{0.x}.{0.y}".format(_t)
                f.write(tile_name + "\n")

    # free the variable for gc
    del feats

    pal = palette(cfg.tags, os.path.join(workspace.other, "colors"))

    cocoPath = os.path.join(workspace.anno, "geococo.json")
    with open(cocoPath, "w", encoding="utf-8") as f:
        with geococo(cfg) as coco:
            for imgIdx, tile in tqdm(enumerate(tile_feats)):
                feats = tile_feats[tile]

                # store geojson
                fc = FeatureCollection(feats)
                tile_name = "{0.z}.{0.x}.{0.y}".format(tile)
                tile_path = os.path.join(tile_dir, tile_name + ".geojson")
                img_path = os.path.join(
                    img_dir, tile_name + ".png"
                )  # default image extension of .png
                with open(tile_path, "w", encoding="utf-8") as gj:
                    try:
                        geojson.dump(fc, gj)
                    except Exception:
                        print("{}.geojson dump wrong!".format(tile_name))
                        assert 0

                # burn tile
                img = {}
                img["id"] = imgIdx
                img["width"] = nx
                img["height"] = ny
                img["file_name"] = tile_name + ".png"
                coco.imgs.append(img)
                geoms = check_topo(feats, tile, nx, ny)
                burned_feats = burn_tile(geoms, cfg.task, pal, img_path, nx, ny)
                base_idx = len(coco.annos)
                for idx, label, coords in burned_feats:
                    catIdx = coco.catIdxs[label]
                    try:
                        coco.annos.append(
                            gen_anno(coords, base_idx + idx, imgIdx, catIdx)
                        )
                    except Exception:
                        print(imgIdx)

            json.dump(coco.to_json(), f, indent=2)


def _overlap(main_geojson, sub_geojson, output_geojson, label):
    with open(main_geojson, "r") as f:
        main_data = json.load(f)
    main_geometries = [shape(feature["geometry"]) for feature in main_data["features"]]

    with open(sub_geojson, "r") as f:
        sub_data = json.load(f)
    sub_geometries = [shape(feature["geometry"]) for feature in sub_data["features"]]

    index_tree = STRtree(sub_geometries)

    res_feats = []

    for main_geom in main_geometries:
        intersecting_geometries = index_tree.query(main_geom)
        for sub_geom in intersecting_geometries:
            if main_geom.intersects(sub_geom):
                # feat = Feature(geometry=main_geom, properties={"label": label})
                feat = Feature(geometry=main_geom)
                res_feats.append(feat)
                break
    fc = FeatureCollection(res_feats)

    with open(output_geojson, "w") as f:
        geojson.dump(fc, f)
