import io
import json

# Create data file in decathlon format
def create_json_data_file(
        train_files,
        output_file_path,
        train_masks=(),
        val_files=(),
        val_masks=(),
        train_key='training',
        val_key='validation'
        ):

    # Create dictionary
    files_in_decathlon_format = {}
    if len(train_masks) > 0:
        files_in_decathlon_format[train_key] = [{'image': path, 'label': mask} for path, mask in zip(train_files, train_masks)]
        if len(val_files) > 0:
            files_in_decathlon_format[val_key] = [{'image': path, 'label': mask} for path, mask in zip(val_files, val_masks)]
    else:
        files_in_decathlon_format[train_key] = [{'image': path} for path in train_files]
        if len(val_files) > 0:
            files_in_decathlon_format[val_key] = [{'image': path} for path in val_files]

    # Write json file
    with io.open(output_file_path, 'w', encoding='utf8') as output_file:
        json.dump(files_in_decathlon_format, output_file, indent=4, ensure_ascii=False)



