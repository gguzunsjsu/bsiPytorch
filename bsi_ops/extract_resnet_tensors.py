import torch
import torchvision.models as models
print('import works')

def get_fc_weight_row_one():
    model = models.resnet50(pretrained=True)
    # Create a dummy input tensor with the required shape
    dummy_input = torch.randn(1, 3, 224, 224)  # (batch_size, channels, height, width)

    with torch.no_grad():
        model.eval()
        data_vectors = []
        for name, layer in model.named_children():
            if name == 'avgpool':
                break  # Stop after the avgpool layer to avoid fully connected layers
            dummy_input = layer(dummy_input)
            data_vectors.append(dummy_input.cpu().numpy())

    #for i, data_vector in enumerate(data_vectors):
    #    print(f"Layer {i}: Shape {data_vector.shape}")

    weight_vectors = {}
    #print("\nWeight Vectors (Parameters):")
    for name, param in model.named_parameters():
        if 'weight' in name:
            #print(f'Layer: {name}, Shape: {param.shape}')
            weight_vectors[name] = param.detach().cpu().numpy()
    #print(weight_vectors["fc.weight"][1])
    #print(weight_vectors["fc.weight"][1].shape)
    return weight_vectors["fc.weight"][1]