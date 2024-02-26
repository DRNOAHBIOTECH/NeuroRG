class ResultUtils:
    def __init__(self):
        pass

    @staticmethod
    def compute_ratios(group):
        # Count the number of predictions for each value
        counts = group['prediction'].value_counts()

        # Compute the ratios for each value
        ratios = counts / counts.sum()

        # Create a new column for the prediction_well value
        prediction_well = ratios.idxmax()

        d = {}
        for i in range(6):
            col = str(i)
            d[col] = ratios.get(i, 0)

        d['prediction'] = prediction_well
        d['label'] = group['label'].value_counts().idxmax()
        d['check'] = (d['prediction'] == d['label'])
        return pd.Series(d, index=['label', 'prediction', '0', '1', '2', '3', '4', '5', 'check'])
    
    @staticmethod
    def evaluate_model(model, dataloaders_test, device):
        model.eval()
        prediction = []
        labels_list = []
        correct = 0
        total = 0

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for images, labels in dataloaders_test:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    prediction.append(predicted.cpu().numpy())
                    labels_list.append(labels.cpu().numpy())
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test images: {accuracy} %')

        return accuracy, np.concatenate(prediction), np.concatenate(labels_list)

    @staticmethod
    def save_results(well_df, model_name, module_name, day, plate_idx, channel, Target_addr):
        well_df['model'] = model_name
        well_df['day'] = day
        well_df['module'] = module_name
        well_df['phase'] = 'train'
        well_df['plate_idx'] = plate_idx
        well_df['val_plates'] = '_'.join(val_plate_list)
        well_df['channel'] = channel_dict[channel]

        # Save image predictions
        image_csv_path = f'{Target_addr}/Val_predicted_image_{model_name}_{module_name}_{day}_val_{plate_idx}_ch_{channel}.csv'
        well_df.to_csv(image_csv_path)

        # Apply the function to each group and combine the results
        result = well_df[['module', 'phase', 'day', 'model', 'plate_idx', 'val_plates', 'class', 'well', 'prediction', 'label', 'channel']].groupby(['module', 'phase', 'day', 'model', 'plate_idx', 'val_plates', 'class', 'well', 'channel']).apply(compute_ratios)

        # Calculate and save class-wise ratios
        num_of_class = len(severe_list_sorted)
        acc_prediction = sum(result['check']) / len(result) * 100
        print(f'Prediction accuracy for Wells: {acc_prediction}')

        result_csv_path = f'{Target_addr}/Val_predicted_result_{model_name}_{module_name}_{day}_val_{plate_idx}_ch_{channel}.csv'
        result.to_csv(result_csv_path)