def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the model on test data."""
    model.eval()
    test_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            test_loss += loss.item()
            
            preds = torch.argmax(output, dim=1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    logger.info('Test Loss: %.4f', avg_test_loss)
    mlflow.log_metric("test_loss", avg_test_loss)

    return avg_test_loss, y_true, y_pred