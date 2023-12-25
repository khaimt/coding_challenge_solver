from prompt_template.code_llama_template import CodellamaTemplate, PromptTemplate
from typing import List, Any

name_to_templates = {}


def _register_prompt_template(template_class: Any):
    global name_to_templates
    template_ob = template_class.get_prompt_template()
    name_to_templates[template_ob.name] = template_ob


_register_prompt_template(CodellamaTemplate)


def get_all_template_names() -> List[str]:
    return list(name_to_templates.keys())


def get_prompt_template_by_name(name: str) -> PromptTemplate:
    if name in name_to_templates:
        return name_to_templates[name]

    raise Exception(f"template not found: {name}")
