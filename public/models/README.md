Place these matching model files here before building or deploying:
- signvision_model.onnx
- labels.json
- model_metadata.json

Required input: float32 [1,63], MediaPipe's 21 XYZ landmarks normalized relative to the wrist and divided by the maximum Euclidean wrist distance. See the root README for required metadata fields.
