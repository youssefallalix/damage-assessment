import rasterio
import os


def gdal_translate(input_file):
    output_file = f"{os.path.splitext(input_file)[0]}_processed.tif"

    with rasterio.open(input_file) as src:
        bands = src.read()

        profile = src.profile
        profile.update({
            'compress': 'lzw',
            'predictor': 2,
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256
        })

        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(bands)

    os.remove(input_file)
    return output_file
