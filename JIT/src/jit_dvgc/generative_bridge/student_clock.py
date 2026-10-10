"""Opt-in fixed Actor coordinates with an actual completed rollout clock."""
import ast
import inspect


def demo_weight(completed_transitions,budget,start,end):
    import jax.numpy as jp
    return start+(end-start)*jp.clip(jp.asarray(completed_transitions)/budget,0.,1.)


def attach_clock(data,env_steps,rollout_transitions):
    import jax.numpy as jp
    from ..policy_distillation import _count_float
    completed=_count_float(env_steps)+rollout_transitions
    return data._replace(extras={**data.extras,'policy_extras':{**data.extras['policy_extras'],
        'completed_training_transitions':jp.full(data.discount.shape,completed)}})


def instrument_frozen_normalizer(trainer):
    """Keep the whole copied statistics state; fail closed on upstream drift.

    Clock is sampled physical transitions completed before the current SGD,
    not normalizer.count, optimizer loss calls, or supervised examples. Resume
    uses the declared stage-local env_steps. Existing round continuation resets
    that clock at the new stage; no lifetime-normalizer clock is inferred.
    """
    source=getattr(trainer,'_jit_source_text',None) or inspect.getsource(trainer)
    tree=ast.parse(source);counts=[0,0]
    class Hooks(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='normalizer_params':
                if ast.unparse(node.value)=='training_state.normalizer_params':
                    counts[0]+=1
                    return [node,*ast.parse('data = _repair_attach_clock(data, training_state.env_steps, env_step_per_training_step)').body]
                if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='running_statistics.update':
                    counts[1]+=1
                    return ast.copy_location(ast.Assign(targets=node.targets,value=ast.Name(id='normalizer_params',ctx=ast.Load())),node)
            return self.generic_visit(node)
    tree=Hooks().visit(tree)
    if counts!=[1,2]:raise ValueError('unsupported pinned PPO normalization layout')
    namespace={**trainer.__globals__,'_repair_attach_clock':attach_clock}
    tree=ast.fix_missing_locations(tree);exec(compile(tree,'<frozen normalizer completed-transition PPO>','exec'),namespace)
    wrapped=namespace[trainer.__name__];wrapped._jit_source_text=ast.unparse(tree)
    return wrapped
