import logging

from .tvqa_eval import do_tvqa_evaluation


def voc_evaluation(dataset, predictions, output_folder, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    logger.info("performing tvqa detection evaluation.")
    return do_tvqa_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
