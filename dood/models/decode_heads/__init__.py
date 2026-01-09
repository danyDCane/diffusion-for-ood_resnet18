try:
    from .segmenter_mask_head import SegmenterMaskTransformerHead
except ImportError:
    # mmseg not installed, SegmenterMaskTransformerHead not available
    SegmenterMaskTransformerHead = None