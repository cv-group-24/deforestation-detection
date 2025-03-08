from data.dataset import ForestNetDataset


def main():
    dataset_path  = r"C:\Users\chris\Desktop\University\Code\ComputerVision\ForestNetDataset"
    test_data_set = ForestNetDataset(df = dataset_path, dataset_path=dataset_path)
    print(test_data_set.get_osm_features(-0.23142888714857, 99.87426629402542, False))

 
if __name__ == "__main__":
    main()