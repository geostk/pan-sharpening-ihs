from module import GDM
from module import TRM
import datetime
import example

# 4 3 2 (red green blue pan)


def start():
    start_options = int(input("Welche Datei soll geladen werden?" + "\n 1: klein \n 2: mittel \n 3: groß \n 0: Beenden"))

    if start_options == 1:
        importer = GDM.GeoTifManager()
        do_the_rest(importer)
    elif start_options == 2:
        importer = GDM.GeoTifManager("medium/b4_resize1.tif", "medium/b3_resize1.tif", "medium/b2_resize1.tif", "medium/b8_resize1.tif")
        do_the_rest(importer)
    elif start_options == 3:
        importer = GDM.GeoTifManager("large/b4_resize0.TIF", "large/b3_resize0.TIF", "large/b2_resize0.TIF", "large/b8_resize0.TIF")
        do_the_rest(importer)
    elif start_options == 0:
        exit(0)

    else:
        print("Falsche eingabe")
        start()

def do_the_rest(importer):
    normalized_red, normalized_green, normalized_blue, normalized_panchromatic = importer.load_images()
    transformer = TRM.TransformationManager(normalized_red, normalized_green, normalized_blue, normalized_panchromatic)

    # Mittelwert - Test on CPU
    starttime = datetime.datetime.now()
    red, green, blue = transformer.arithmetic_average()
    print("Duration arithmetic average on CPU: " + str(datetime.datetime.now() - starttime))
    importer.save_images(red, green, blue, "A-M-CPU-")
#
    # Mittelwert - Test on GPU
    starttime = datetime.datetime.now()
    red, green, blue = transformer.arithmetic_average_gpu()
    print("Duration arithmetic average on GPU: " + str(datetime.datetime.now() - starttime))
    importer.save_images(red, green, blue, "A-M-GPU-")

    # IHS Transfomration (Matrix - CPU)
    starttime = datetime.datetime.now()
    red, green, blue = transformer.matrix()
    print("Duration IHS Transfomration on CPU with matrix: " + str(datetime.datetime.now() - starttime))
    importer.save_images(red, green, blue, "Matrix-CPU-")

    # IHS Transfomration (Matrix - GPU)
    starttime = datetime.datetime.now()
    red, green, blue = transformer.matrix_gpu()
    print("Duration IHS Transfomration on GPU with matrix: " + str(datetime.datetime.now() - starttime))
    importer.save_images(red, green, blue, "Matrix-GPU-")

    # IHS Transfomration (Matrix - GPU)
    #starttime = datetime.datetime.now()
    #red, green, blue = transformer.matrix_cpu_c_module()
    #print("Duration IHS Transfomration on CPU with matrix: " + str(datetime.datetime.now() - starttime))
    #importer.save_images(red, green, blue, "Matrix-CPU-C-Modul-")

    # # IHS Transfomration - Test on GPU
    # starttime = datetime.datetime.now()
    # intensity, hue, saturation = transformer.rgb_to_ihs_gpu()
    # new_intensity = transformer.intensity_mixer_gpu(intensity, 0.3, 0.7)
    # red, green, blue = transformer.ihs_to_rgb_cpu(new_intensity, hue, saturation)
    # print("Duration IHS Transfomration on GPU: " + str(datetime.datetime.now() - starttime))
    # importer.save_images(red, green, blue, "IHS-GPU-")
    #
    #
    # # IHS Transfomration - Test on CPU
    # starttime = datetime.datetime.now()
    # intensity, hue, saturation = transformer.rgb_to_ihs_cpu()
    # new_intensity = transformer.intensity_mixer_cpu(intensity, 0.3, 0.7)
    # red, green, blue = transformer.ihs_to_rgb_cpu(new_intensity, hue, saturation)
    # print("Duration IHS Transfomration on CPU: " + str(datetime.datetime.now() - starttime))
    # importer.save_images(red, green, blue, "IHS-CPU-")

start()
