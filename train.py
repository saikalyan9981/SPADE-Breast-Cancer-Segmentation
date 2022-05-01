from segmentation_models_pytorch.utils.train import Epoch
import sys
import torch
from tqdm import tqdm as tqdm
from segmentation_models_pytorch.utils.meter import AverageValueMeter
class SpadeEpoch(Epoch):
    def batch_update(self, x, y, c):
        raise NotImplementedError 
    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y, c in iterator:
            #for x, y in iterator:
                # print("x = ", x)
                # print("y = ", y)
                # print("c = ", c)
                x, y, c = x.to(self.device), y.to(self.device), c.to(self.device)
                #x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y, c)
                #loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs
class TrainSpadeEpoch(SpadeEpoch):
    def __init__(self, model, loss, metrics, optimizer,scheduler=None, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        #self.test_imlg = torch.load(r'Q:\TIGER\bimglg.pt', map_location=torch.device(device))

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, c):
    #def batch_update(self, x, y):

        self.optimizer.zero_grad()
        # print(f"x.shape = {x.shape};  c.shape = {c.shape}")
        prediction = self.model.forward((x, c))
        #prediction = self.model.forward(x)
        #print('model weight', self.model.state_dict())
        #print("Prediction = ", prediction)
        loss = self.loss(prediction, y)
        # print("y = ", y)
        # print("loss = ", loss)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidSpadeEpoch(SpadeEpoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )
        #self.test_imlg = torch.load(r'Q:\TIGER\bimglg.pt', map_location=torch.device(device))

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, c):
    #def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward((x, c))
            #prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction