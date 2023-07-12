import argparse
import os
from shapely.geometry import shape
from shapely.strtree import STRtree
import geojson
from geojson import Feature, FeatureCollection
from ohsome2label.config import Config, Parser, workspace

from ohsome2label.label import gen_label
from ohsome2label.utils import download_osm, download_img


# 加载建筑物图层

def overlap(main_geojson, sub_geojson, output_geojson,label):
    with open(main_geojson, 'r') as f:
        main_data = geojson.load(f)
    main_geometries = [shape(feature['geometry']) for feature in main_data['features']]

    with open(sub_geojson, 'r') as f:
        sub_data = geojson.load(f)
    sub_geometries = [shape(feature['geometry']) for feature in sub_data['features']]

    index_tree = STRtree(sub_geometries)

    res_feats = []
    
    for main_geom in main_geometries:
        intersecting_geometries = index_tree.query(main_geom)
        for sub_geom in intersecting_geometries:
            if main_geom.intersects(sub_geometries[sub_geom]):
                feat = Feature(geometry=main_geom, properties={"label" : label})
                res_feats.append(feat)
                break
    fc = FeatureCollection(res_feats) 

    with open(output_geojson, "w") as f:
        geojson.dump(fc, f)

def main(config):
    schema = os.path.split(config)[0] + '/schema.yaml'
    o2l_cfg = Parser(config, schema).parse()
    wspace = workspace(o2l_cfg.workspace)

    print(o2l_cfg.tags)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='logistics_scripts')
    parser.add_argument('config', type=str, help='config_path')

    args = parser.parse_args()

    main(args.config)