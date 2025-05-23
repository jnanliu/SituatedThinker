from .retrieval.interface import Retrieval
from .code.interface import CodeExecution
from .webqsp_freebase.interface import RelationRetrieval, TailEntityRetrieval
from .wtq_table.interface import Headers, Column, Row
from .textworld_game.interface import Feedback, Description, AdmissibleCommands, PossibleAdmissibleCommands

from .base import InterfaceZoo


_lazy_import = {
    "retrieval_and_code": lambda: InterfaceZoo.from_interface_cls_list([Retrieval, CodeExecution]),
    "webqsp": lambda: InterfaceZoo.from_interface_cls_list([RelationRetrieval, TailEntityRetrieval]),
    "wtq": lambda: InterfaceZoo.from_interface_cls_list([Headers, Column, Row]),
    "textworld": lambda: InterfaceZoo.from_interface_cls_list([Feedback, Description, AdmissibleCommands, PossibleAdmissibleCommands])
}
def __getattr__(name):
    if name in globals():
        return globals()[name]
    else:
        if name in _lazy_import:
            globals()[name] = _lazy_import[name]()
            return globals()[name]

    raise AttributeError(f"module 'interfaces' has no attribute '{name}'")