from src.preprocess import run_preprocessing
from src.train import run_training
from src.utils import get_logger

log = get_logger(__name__)


def run_pipeline() -> None:
    log.info("Pipeline başlıyor: preprocess -> train")
    run_preprocessing()
    run_training()
    log.info("Pipeline tamamlandı.")


if __name__ == "__main__":
    run_pipeline()
