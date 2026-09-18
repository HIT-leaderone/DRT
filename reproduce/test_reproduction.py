"""CPU-only checks for portable release configuration and documentation."""

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


class ReproductionTests(unittest.TestCase):
    def test_rl_recipe_composes(self):
        with patch.dict(os.environ, {
            'DRT_REPO_ROOT': str(ROOT),
            'DRT_DATA_DIR': '/data/drt',
            'DRT_SFT_MODEL': 'leaderonehit/DRT-SFT-8B',
            'DRT_REWARD_MODEL': 'Qwen/Qwen3-235B-A22B-Instruct-2507',
            'DRT_OUTPUT_DIR': './checkpoints/drt-rl',
        }):
            with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'reproduce/configs')):
                cfg = OmegaConf.to_container(compose(config_name='drt_rl'), resolve=True)
        actor = cfg['actor_rollout_ref']['actor']
        rollout = cfg['actor_rollout_ref']['rollout']
        reward = cfg['reward']
        self.assertEqual(cfg['data']['train_files'], '/data/drt/train.parquet')
        self.assertEqual(cfg['data']['train_batch_size'], 512)
        self.assertEqual(actor['ppo_mini_batch_size'], 128)
        self.assertEqual(actor['optim']['lr'], 1e-6)
        self.assertEqual(actor['kl_loss_coef'], 0.01)
        self.assertTrue(actor['use_kl_loss'])
        self.assertEqual(actor['megatron']['tensor_model_parallel_size'], 2)
        self.assertEqual(actor['megatron']['pipeline_model_parallel_size'], 2)
        self.assertEqual(rollout['n'], 16)
        self.assertEqual(rollout['tensor_model_parallel_size'], 4)
        self.assertTrue(reward['reward_model']['enable_resource_pool'])
        self.assertEqual(reward['reward_model']['rollout']['tensor_model_parallel_size'], 8)
        kwargs = reward['custom_reward_function']['reward_kwargs']
        self.assertEqual(kwargs['step_bonus_lambda'], 0.3)
        self.assertEqual(kwargs['process_gamma'], 1.5)
        self.assertEqual(kwargs['deep_exploration_difficulty_by_source']['vision_r1_rl'],
                         {'threshold': 6, 'cap': 9})
        self.assertEqual(kwargs['deep_exploration_difficulty_by_source']['dapo_math_17k'],
                         {'threshold': 10, 'cap': 18})
        self.assertTrue(Path(reward['custom_reward_function']['path']).is_file())
        self.assertIsNone(cfg['trainer']['default_hdfs_dir'])
        self.assertFalse(cfg['trainer']['auto_eval']['enabled'])

    def test_eval_recipe_uses_released_models_and_real_classes(self):
        cfg = json.loads((ROOT / 'reproduce/configs/eval.json').read_text())
        self.assertEqual(set(cfg['model']), {'DRT-SFT-8B', 'DRT-RL-8B'})
        for name, model in cfg['model'].items():
            self.assertEqual(model['model_path'], 'leaderonehit/' + name)
            self.assertEqual(model['class'], 'Qwen3VLChat')
            self.assertEqual(model['temperature'], 0.7)
            self.assertEqual(model['max_new_tokens'], 16384)
            self.assertTrue(model['use_custom_prompt'])
        self.assertEqual(set(cfg['data']), {'MathVista_MINI', 'MathVerse_MINI', 'LogicVista',
                                           'GSM8K', 'Video_Holmes'})
        classes = set()
        for path in (ROOT / 'VLMEvalkit/vlmeval/dataset').glob('*.py'):
            classes.update(n.name for n in ast.walk(ast.parse(path.read_text()))
                           if isinstance(n, ast.ClassDef))
        for dataset in cfg['data'].values():
            self.assertIn(dataset['class'], classes)
        self.assertEqual(cfg['data']['Video_Holmes']['nframe'], 32)

    def test_document_links_and_code_syntax(self):
        for name in ['README.md', 'reproduce/README.md', 'reproduce/environment.md',
                     'training_data/README.md']:
            path = ROOT / name
            content = path.read_text()
            for target in re.findall(r'\]\(([^)]+)\)', content):
                if target.startswith(('http:', 'https:', '#')):
                    continue
                self.assertTrue((path.parent / target.split('#')[0]).exists(), (name, target))
            for language, code in re.findall(r'```(\w+)\n(.*?)```', content, re.S):
                if language == 'python':
                    ast.parse(code)
                elif language == 'bash':
                    result = subprocess.run(['bash', '-n'], input=code, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, (name, result.stderr))

    def test_data_manifest_totals(self):
        manifest = json.loads((ROOT / 'training_data/rl_training_data_manifest.json').read_text())
        self.assertEqual(manifest['total_size_bytes'], sum(f['size_bytes'] for f in manifest['files']))
        self.assertEqual([f['rows'] for f in manifest['files'][:3]], [14724, 618, 32362])
        for item in manifest['files']:
            self.assertRegex(item['sha256'], r'^[a-f0-9]{64}$')
        self.assertRegex(manifest['huggingface_revision'], r'^[a-f0-9]{40}$')


if __name__ == '__main__':
    unittest.main()
