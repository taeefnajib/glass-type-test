import sklearn
import os
import typing
from flytekit import workflow
from project.wf.main import Hyperparameters
from project.wf.main import run_wf

_wf_outputs=typing.NamedTuple("WfOutputs",run_wf_0=sklearn.neighbors._classification.KNeighborsClassifier)
@workflow
def wf_10(_wf_args:Hyperparameters)->_wf_outputs:
	run_wf_o0_=run_wf(hp=_wf_args)
	return _wf_outputs(run_wf_o0_)