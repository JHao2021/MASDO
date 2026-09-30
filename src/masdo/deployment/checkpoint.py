"""Export only the frozen skill library, assignment network, and actor."""
from copy import deepcopy

import torch

from masdo.assignment.policy import VectorSkillExecutor
from masdo.orchestration.policy import SkillOrganizer

ENVIRONMENT_KEYS = ('coordinate_scale_km', 'travel_speed', 'd_ref', 'max_tasks', 'encoder_max_tasks', 'max_task_value',
                    'grid_bounds', 'grid_rows', 'grid_cols')
SCHEMA_KEYS = {'format', 'environment', 'executor_config', 'executor', 'organizer_config', 'organizer'}
DEPLOYMENT_FORMAT = 'MASDO'


class DeployedVectorExecutor(VectorSkillExecutor):
    def __init__(self, **config):
        super().__init__(**config)
        del self.auxiliary_offset

    def auxiliary_value(self, *args, **kwargs):
        raise RuntimeError('No auxiliary training value in deployment')


def deployment_bundle(executor, organizer, environment):
    if not torch.equal(executor.embedding.weight.detach().cpu(), organizer.skill_embeddings.detach().cpu()):
        raise ValueError('Organizer must use the frozen assignment skill library')
    return dict(format=DEPLOYMENT_FORMAT,
        environment={k: deepcopy(environment[k]) for k in ENVIRONMENT_KEYS if k in environment},
        executor_config=deepcopy(executor.config),
        executor={k: v.detach().cpu().clone() for k, v in executor.state_dict().items()
                  if not k.startswith('auxiliary_offset.')},
        organizer_config=dict(executor_hidden=organizer.executor_hidden, hidden=organizer.hidden),
        organizer={k: v.detach().cpu().clone() for k, v in organizer.state_dict().items()
                   if not k.startswith('critic.')})


def load_deployment(bundle, device='cpu'):
    if set(bundle) != SCHEMA_KEYS or bundle['format'] != DEPLOYMENT_FORMAT:
        raise ValueError('Invalid deployment checkpoint')
    if set(bundle['environment']) - set(ENVIRONMENT_KEYS):
        raise ValueError('Invalid environment configuration')
    if any(k.startswith('auxiliary_offset.') for k in bundle['executor']) or any(
            k.startswith('critic.') for k in bundle['organizer']):
        raise ValueError('Deployment must omit training-only values')
    executor = DeployedVectorExecutor(**bundle['executor_config']).to(device)
    executor.load_state_dict(bundle['executor'], strict=True)
    organizer = SkillOrganizer(executor.embedding.weight, **bundle['organizer_config'], with_critic=False).to(device)
    organizer.load_state_dict(bundle['organizer'], strict=True)
    if not torch.equal(executor.embedding.weight, organizer.skill_embeddings):
        raise ValueError('Exported skill libraries differ')
    executor.eval().requires_grad_(False)
    organizer.eval().requires_grad_(False)
    return executor, organizer, deepcopy(bundle['environment'])
