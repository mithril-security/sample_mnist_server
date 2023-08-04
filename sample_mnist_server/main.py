import uvicorn
from fastapi import FastAPI
import torch
from torchvision.models.resnet import resnet18

from batch_runner import BatchRunner
from collators import TorchCollator
from serializers import async_array_endpoint


app = FastAPI()


model = resnet18()
weights = torch.load("./model.pth")
model.load_state_dict(weights)


runner = BatchRunner(
    model,
    max_batch_size=256,
    max_latency_ms=200,
    collator=TorchCollator(),
)
app.on_event("startup")(runner.run)


@app.post("/predict")
@async_array_endpoint(sample_rate=16000)
async def predict(x: torch.Tensor) -> torch.Tensor:
    return await runner.submit(x)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
