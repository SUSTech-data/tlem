from evaluate import EvaluationSuite
from tlem.tasks import fake_pipeline
from ipytorch import logging

from tlem import Suite


def test_huggingfacce():
    suite = EvaluationSuite.load("sustech/tlem", download_mode="force_redownload")
    results = suite.run(fake_pipeline)
    logging.info(results)


def test_local():
    results = Suite(name="tlem").run(fake_pipeline)
    logging.info(results)
