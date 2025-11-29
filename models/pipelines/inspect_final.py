from .final_pipeline import FinalPipeline

if __name__ == '__main__':
    pipeline = FinalPipeline.load_checkpoint(
        './.checkpoints/FinalPipeline/_epoch=1_step=580_loss=1.4083.pt'
    )
    pipeline.inspect_weight()