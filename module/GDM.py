import gdal
import numpy as np
import datetime
from gdalconst import *
import struct
import osr




class GeoTifManager:
    def __init__(self,
                 path_to_red_image="small/b4_resize.tif",
                 path_to_green_image="small/b3_resize.tif",
                 path_to_blue_image="small/b2_resize.tif",
                 path_to_panochrome_image="small/b8.tif"):

        self.path_to_red_image = path_to_red_image
        self.path_to_green_image = path_to_green_image
        self.path_to_blue_image = path_to_blue_image
        self.panchromatic = path_to_panochrome_image

        self.dataset = gdal.Open(self.path_to_red_image)

    def load_images(self):

        def transform_to_a_normalized_2d_numpy_array(image):
            starttime = datetime.datetime.now()
            band = image.GetRasterBand(1)
            numpyarray = np.empty([band.YSize, band.XSize], dtype=np.uint16)

            if image == red_image:
                print("Lade roten Kanal: ")
            elif image == green_image:
                print("Lade grünen Kanal: ")
            elif image == blue_image:
                print("Lade blauen Kanal: ")
            elif image == pan_image:
                print("Lade Panochromen Kanal: ")
            else:
                print("Loading unknown image...")

            for row in range(0, image.RasterYSize):
                if row % 3000 == 0:
                    print(str(row) + ": " + str(datetime.datetime.now() - starttime))
                scanline = band.ReadRaster(0, row, band.XSize, 1, band.XSize, 1, GDT_Float32)
                starttime = datetime.datetime.now()
                # Umwandel von "xsize*4 bytes of raw binary floating point data" zu Werten zwischen 0 und 255
                numpyarray[row] = struct.unpack('f' * band.XSize, scanline)

            # Es werden die normalisierten(0 - 1) Farb - Werte in eine 2 D Array zurück gegeben.
            return numpyarray  # / float(255)

        red_image = gdal.Open(self.path_to_red_image)
        green_image = gdal.Open(self.path_to_green_image)
        blue_image = gdal.Open(self.path_to_blue_image)
        pan_image = gdal.Open(self.panchromatic)

        if red_image is None:
            quit(str("Quit.. red band missing"))
        else:
            print("Red band loaded")

        if green_image is None:
            quit(str("Quit.. green band missing"))
        else:
            print("Green band loaded")

        if blue_image is None:
            quit(str("Quit.. blue band missing"))
        else:
            print("Blue band loaded")

        if pan_image is None:
            quit(str("Quit.. panchromatic band missing"))
        else:
            print("Panchromatic band loaded")

        normalized_numpy_2d_array_red = transform_to_a_normalized_2d_numpy_array(red_image)
        normalized_numpy_2d_array_green = transform_to_a_normalized_2d_numpy_array(green_image)
        normalized_numpy_2d_array_blue = transform_to_a_normalized_2d_numpy_array(blue_image)
        normalized_numpy_2d_array_pan = transform_to_a_normalized_2d_numpy_array(pan_image)

        return normalized_numpy_2d_array_red, normalized_numpy_2d_array_green, normalized_numpy_2d_array_blue, normalized_numpy_2d_array_pan

    def save_images(self, new_red_image, new_green_image, new_blue_image, prefix):
        # Dateiformat der Output-Dateien
        driver = gdal.GetDriverByName("GTiff")

        # Geoinformationen
        geotransform = self.dataset.GetGeoTransform()
        # Banindormationen (Größe, etc)
        bandinfomration = self.dataset.GetRasterBand(1)

        # Zusatzinfos wie Projection
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        srs.SetUTM(32, 1)
        srs.SetWellKnownGeogCS('WGS84')

        # Erstellen einer neuen Datei:
        dst_ds_red = driver.Create("output/" + prefix + "pan_sharped_red.tif", bandinfomration.XSize, bandinfomration.YSize, 1,
                                   gdal.GDT_UInt16, ['COMPRESS=LZW'])

        # Infomationen über die Aausprägung, Pixelgröße...
        dst_ds_red.SetGeoTransform([geotransform[0], geotransform[1], 0, geotransform[3], 0, geotransform[5]])

        # Speichern der neuen Bildinformationen in Band 1
        outerband = dst_ds_red.GetRasterBand(1)
        outerband.WriteArray(new_red_image)
        dst_ds_red.SetProjection(srs.ExportToWkt())
        # --------------------------------------------------------------------------------------------------
        dst_ds_green = driver.Create("output/" + prefix + "pan_sharped_green.tif", bandinfomration.XSize, bandinfomration.YSize, 1,
                                     gdal.GDT_UInt16, ['COMPRESS=LZW'])
        dst_ds_green.SetGeoTransform([geotransform[0], geotransform[1], 0, geotransform[3], 0, geotransform[5]])

        # Speichern der neuen Bildinformationen in Band 2
        outerband = dst_ds_green.GetRasterBand(1)
        outerband.WriteArray(new_green_image)
        dst_ds_green.SetProjection(srs.ExportToWkt())
        # --------------------------------------------------------------------------------------------------
        dst_ds_blue = driver.Create("output/" + prefix + "pan_sharped_blue.tif", bandinfomration.XSize, bandinfomration.YSize, 1,
                                    gdal.GDT_UInt16, ['COMPRESS=LZW'])
        dst_ds_blue.SetGeoTransform([geotransform[0], geotransform[1], 0, geotransform[3], 0, geotransform[5]])

        # Speichern der neuen Bildinformationen in Band 3
        outerband = dst_ds_blue.GetRasterBand(1)
        outerband.WriteArray(new_blue_image)
        dst_ds_blue.SetProjection(srs.ExportToWkt())

        outerband.FlushCache()
        dst_ds_red = None
        dst_ds_green = None
        dst_ds_blue = None

        # import gdal_merge as gm
        # sys.path.append('C:/Python27/ArcGIS10.2/Scripts/')
        #
        # workspace = "D:/Satellitendaten/rapideye/img/testregion/cannyedge/out/"
        # os.chdir(workspace)
        #
        # sys.argv[1:] = ['-o', 'out.tif', 'allre1.tif', etc...]

        return 0



