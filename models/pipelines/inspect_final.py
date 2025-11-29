from .final_pipeline import FinalPipeline

if __name__ == '__main__':
    pipeline = FinalPipeline.load_checkpoint(
        './.checkpoints/FinalPipeline/_epoch=2_step=870_loss=1.2584.pt'
    )
    pipeline.inspect_weight()