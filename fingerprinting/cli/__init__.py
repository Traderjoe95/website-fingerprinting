from logging import NOTSET, DEBUG, INFO

from click import group, option, pass_context, Group

from ..api.plugin import ApplicationContext, load_core_modules, install
from ..util.config import CONFIG
from ..util.logging import configure


def load_main() -> Group:
    app = ApplicationContext()

    @group()
    @option(
        "-v",
        count=True,
        help="""Set logging to a higher level (-vv or -vvv for even more logging)"""
    )
    @pass_context
    def main(ctx, v):
        # ensure that ctx.obj exists and is a dict
        ctx.ensure_object(dict)
        ctx.obj['application_ctx'] = app

        if v >= 3:
            configure(NOTSET)
        elif v == 2:
            configure(DEBUG)
        elif v == 1:
            configure(INFO)
        else:
            configure()

    @main.group(help="Manage datasets")
    def dataset():
        pass

    app.set_root(main)

    load_core_modules(app)

    for mod in CONFIG.get_obj("modules").get_list("import").as_list():
        install(mod, app)

    return main
