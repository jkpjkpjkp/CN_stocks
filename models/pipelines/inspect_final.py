from .final_pipeline import FinalPipeline, FinalPipelineConfig


def init_model():
    config = FinalPipelineConfig(
        debug_data=True,
        debug_ddp=True,
        debug_model=True,
    )
    return FinalPipeline(config)


if __name__ == '__main__':
    # pipeline = FinalPipeline.load_checkpoint(
    #     './.checkpoints/FinalPipeline/_epoch=2_step=870_loss=1.2584.pt'
    # )
    pipeline = init_model()
    pipeline.activate()
    pipeline.inspect_weight()