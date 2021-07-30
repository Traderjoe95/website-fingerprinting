from click import pass_context, command

from ._helper import instances, datasets as __datasets, defenses as __defenses, gen_and_run_pipelines, parse_algorithm
from ..api import EvaluationPipeline
from ..api.evaluation import EvaluationConfig
from ..api.plugin import ApplicationContext, CliModule
from ..util.config import CONFIG, ConfigObject


def plugin_path() -> str:
    return 'core.cli.evaluation'


def cli_module() -> CliModule:
    return CliModule(name='run_cli', top_level_commands=[__run])


@command("run")
@pass_context
def __run(ctx):
    app: ApplicationContext = ctx.obj['application_ctx']
    _instances = instances(CONFIG.get_obj("evaluation"))

    def gen_pipeline(inst: ConfigObject) -> EvaluationPipeline:
        _datasets = __datasets(app, inst)
        _defenses = __defenses(app, inst)

        attacks = [at for it in inst.get_list('algorithms.attacks') for at in parse_algorithm(app.lookup_attack, it)]

        return EvaluationPipeline(
            datasets=_datasets,
            defenses=_defenses,
            attacks=attacks
        )

    gen_and_run_pipelines(_instances, gen_pipeline, EvaluationConfig)
