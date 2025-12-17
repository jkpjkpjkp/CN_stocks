from ..pipelines.final_pipeline import PriceHistoryDataset, FinalPipelineConfig


if __name__ == '__main__':
    config = FinalPipelineConfig()
    dataset = PriceHistoryDataset(config, 'train', '/home/jkp/ssd/pipeline_mmap/')
    y = dataset[15530006]
    
    
    for i, x in enumerate(config.features):
        print(x, y[0][:, i].mean(), y[0][:, i].std())