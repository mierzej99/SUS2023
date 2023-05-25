import data_functions
input_list, enc_list = data_functions.list_of_paths('data/train/train.csv')
# creating paris of phots and encodings (0 - photos in pairs is not the same, 1- photo and encoding are the same picture)
data = data_functions.create_list_of_pairs_and_labels(input_list, enc_list, 1)

files, labels = data
sizes = [8, 16, 32, 64]
gray = [True, False]
for s in sizes:
    for gs in gray:
        if not (s==8 and gs == True):
            print(f'start of size={s} and gray={gs}')
            data_functions.load_and_transform_images_scikit(files, gray_scale=gs, size=s)
     
## test   
input_list, enc_list = data_functions.list_of_paths('data/test/test.csv')
# creating paris of phots and encodings (0 - photos in pairs is not the same, 1- photo and encoding are the same picture)
data = data_functions.create_list_of_pairs_and_labels(input_list, enc_list, 0)

files, _ = data
sizes = [8, 16, 32, 64]
gray = [True, False]
for s in sizes:
    for gs in gray:
        print(f'start of size={s} and gray={gs}')
        data_functions.load_and_transform_images_scikit(files, gray_scale=gs, size=s)