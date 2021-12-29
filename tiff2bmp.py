from PIL import Image
import os


def tiff2bmp(input_path, output_path):
    for file in os.listdir(input_path):
        file_path = os.path.join(input_path, file)
        if os.path.isdir(file_path):
            tiff2bmp(file_path, output_path)
        else:
            im = Image.open(file_path)
            for i in range(len(file)):
                if file[-i-1] == '.':
                    break
            output_file = os.path.join(output_path, file[:-i-1]+".bmp")
            im.save(output_file, 'bmp')


if __name__ == '__main__':
    input_path = "dataset"
    output_path = "BMPdataset"
    tiff2bmp(input_path, output_path)
    