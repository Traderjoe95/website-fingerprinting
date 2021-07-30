from typing import List

from ..api.plugin import ApplicationContext
from ..util.config import ConfigObject


def instances(config: ConfigObject) -> List[ConfigObject]:
    defaults = config.get_obj("all")
    _instances = config.get_list("instances")

    return [_instances.get_obj(i).merge_into(defaults) for i in range(len(_instances))]


def datasets(app: ApplicationContext, inst: ConfigObject):
    return [ds for it in inst.get_list('algorithms.datasets') for ds in parse_algorithm(app.lookup_dataset, it)]


def defenses(app: ApplicationContext, inst: ConfigObject):
    d = [df for it in inst.get_list('algorithms.defenses')
         for df in parse_algorithm(app.lookup_defense, it)]

    if inst.get_bool('algorithms.include_no_defense', False):
        d.insert(0, None)

    return d


def parse_algorithm(loader, alg):
    if isinstance(alg, str):
        return loader(alg)
    elif isinstance(alg, ConfigObject):
        name = alg.get_str('name')
        alg = alg.get_str('algorithm')

        loaded = loader(alg)

        if len(loaded) == 1:
            return [(name, loaded[0])]
        else:
            return [(f'{name}-{it.name()}', it) for it in loaded]
    else:
        raise ValueError(f'Invalid type {type(alg)} for algorithm {alg}, expected str or ConfigObject')


def gen_and_run_pipelines(_instances: List[ConfigObject], pipeline_gen, config_type,
                          out_file_default: str = "results.csv"):
    evaluations = [(pipeline_gen(inst), inst.get_obj("settings").as_obj(config_type),
                    inst.get_str("out_file", out_file_default), inst.get_bool("out_append", False))
                   for inst in _instances]

    for pipeline, config, out_file, out_append in evaluations:
        # pass
        results = pipeline.run_evaluation(config)

        if out_append:
            results.to_csv(out_file, mode='a', header=False, index=False)
        else:
            results.to_csv(out_file, mode='w', header=True, index=False)
