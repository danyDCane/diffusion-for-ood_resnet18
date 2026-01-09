try:
    from .distilled_vit import DistilledVisionTransformer
except ImportError:
    # mmseg not installed, VIT models not available
    DistilledVisionTransformer = None