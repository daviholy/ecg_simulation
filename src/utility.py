
from wfdb.io import Record, Annotation, rdrecord, rdann
from typing import Iterator
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy.typing as npt

def calculate_hr(record: Record, ann:Annotation):
    return 60 /(len(record.p_signal)/record.fs) * len([i for i,x in enumerate(ann.symbol) if x =="N"]) # type: ignore

def read_signals(path: Path,sampfrom: int = 0, sampto:int | None = None) -> Iterator[tuple[Annotation, Record]]:
    for header in path.glob("*.hea"):
        ann = rdann(str(header.with_suffix("")), extension="i", return_label_elements=["symbol","label_store","description"])

        sig = rdrecord(header.with_suffix(""),sampfrom=sampfrom,sampto=sampto)
        yield ann, sig

def plot_record(record:Record, generated: npt.NDArray | None = None, height: int=2000, width: int=2000):
    if record.p_signal is None:
        raise Exception("Need to have loaded signal in record")
    
    figure = make_subplots(rows=record.p_signal.shape[1])
    for row_index, signal in enumerate(record.p_signal.T):
        figure.add_trace( go.Scatter(y=signal,x=list(range(record.p_signal.shape[0])), showlegend=False, line={"color": "blue"}),row=row_index + 1, col =1)
        if generated is not None:
            figure.add_trace( go.Scatter(y=generated,x=list(range(record.p_signal.shape[0])), showlegend=False, line={"color": "red"}),row=row_index + 1, col =1)

    figure.update_layout(height=2000,width=2000)
    return figure