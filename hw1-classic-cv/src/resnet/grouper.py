from src.utils.metrics_resnet import extract_features_resnet, build_groups, list_image_paths


def group_images(train_path, test_path, threshold=0.95):
    print("Loading images...")
    all_paths = {}
    # all_paths.update(load_images_from_folder(train_path))
    # all_paths.update(load_images_from_folder(test_path))
    all_paths.update(list_image_paths(train_path))
    all_paths.update(list_image_paths(test_path))

    print("Extracting features...")
    features = extract_features_resnet(all_paths)

    print("Building groups...")
    groups = build_groups(features, threshold=threshold)

    print(f"\nTotal groups found: {len(groups)}")
    for i, group in enumerate(groups):
        print(f"\nGroup {i+1}:")
        for img_name in group:
            print(f"  {img_name}")

    return groups