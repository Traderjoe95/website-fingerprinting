from click import command, pass_context

from ._helper import gen_and_run_pipelines, instances, defenses as __defenses, datasets as __datasets, parse_algorithm
from ..api.plugin import CliModule, ApplicationContext
from ..bounds.bayes_error import ErrorBoundsConfig, ErrorBoundsPipeline
from ..util.config import ConfigObject, CONFIG


def plugin_path() -> str:
    return 'core.cli.error_bound'


def cli_module() -> CliModule:
    return CliModule(name='error_bound_cli', top_level_commands=[__error_bound])


@command("error-bound")
@pass_context
def __error_bound(ctx):
    app: ApplicationContext = ctx.obj['application_ctx']
    _instances = instances(CONFIG.get_obj("error_bound"))

    def gen_pipeline(inst: ConfigObject) -> ErrorBoundsPipeline:
        datasets = __datasets(app, inst)
        defenses = __defenses(app, inst)

        feature_sets = [at for it in inst.get_list('algorithms.feature_sets')
                        for at in parse_algorithm(app.lookup_attack, it)]

        return ErrorBoundsPipeline(
            datasets=datasets,
            defenses=defenses,
            attacks=feature_sets
        )

    gen_and_run_pipelines(_instances, gen_pipeline, ErrorBoundsConfig, "error_bounds.csv")
