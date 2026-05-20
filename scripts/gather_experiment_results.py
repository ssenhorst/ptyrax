#!/usr/bin/env python3
"""
Gather and pool DVC experiment results into a consolidated HDF5 file.

This script collects all experiments from a DVC sweep, extracts their outputs
(models, training data, etc.) and consolidates them into a single HDF5 file
for easy analysis.

The script can find experiments from two sources:
1. Applied/committed experiments (via `dvc exp show`)
2. Queued experiments, even if completed (via git refs in .git/refs/exps/)

This is important because queued DVC experiments don't show up in `dvc exp show`
until they are explicitly applied and committed, even if they've completed
successfully. The script now automatically checks both sources.

Optionally, it can also apply, commit, and push all experiments to git/DVC
remotes using the --push flag. This handles the workflow of saving experiment
sweeps by iteratively applying each experiment in a worktree, committing it,
and pushing to the remote.

Usage:
    # Gather experiment results (works with both applied and queued experiments)
    python scripts/gather_experiment_results.py <sweep_id> [--output output.hdf5]
    python scripts/gather_experiment_results.py --stage <stage_name> [--output output.hdf5]
    
    # Gather and push experiments to remote
    python scripts/gather_experiment_results.py <sweep_id> --push [--remote origin]
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from ruamel.yaml import YAML

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

yaml = YAML()


def filter_stages(
    all_stages: List[str],
    include_stages: Optional[List[str]] = None,
    exclude_stages: Optional[List[str]] = None,
) -> List[str]:
    """Filter stages based on include/exclude lists.
    
    Args:
        all_stages: All available stages
        include_stages: If provided, only include these stages
        exclude_stages: If provided, exclude these stages
        
    Returns:
        Filtered list of stages
    """
    filtered = all_stages.copy()
    
    # Apply include filter (whitelist)
    if include_stages:
        filtered = [s for s in filtered if s in include_stages]
        logging.info(f"Including only stages: {', '.join(filtered)}")
    
    # Apply exclude filter (blacklist)
    if exclude_stages:
        filtered = [s for s in filtered if s not in exclude_stages]
        logging.info(f"Excluding stages: {', '.join(exclude_stages)}")
    
    return filtered


class GitWorktree:
    """Context manager for git worktrees with automatic cleanup.
    
    Ensures worktrees are properly removed even if the script is interrupted
    or encounters errors.
    
    Use this context manager when the worktree is only needed within a single
    function scope. For cases where the worktree must outlive the function
    (e.g., when gathering file paths for later processing), use manual management
    with cleanup_worktree().
    """
    
    def __init__(self, exp_name: str, commit: str, prefix: str = "gather"):
        """Initialize worktree manager.
        
        Args:
            exp_name: Experiment name
            commit: Git commit SHA
            prefix: Prefix for worktree directory name
        """
        self.exp_name = exp_name
        self.commit = commit
        self.worktree_dir = Path(f"/tmp/ptyrax_gather_worktrees/{prefix}_worktree_{exp_name}_{commit[:8]}")
        
    def __enter__(self) -> Path:
        """Create and return worktree path."""
        # Remove existing worktree if present
        cleanup_worktree(self.worktree_dir)
        
        # Create git worktree at the experiment's commit
        logging.debug(f"  Creating worktree at {self.worktree_dir} from commit {self.commit[:8]}")
        try:
            result = subprocess.run(
                ["git", "worktree", "add", "--detach", str(self.worktree_dir), self.commit],
                capture_output=True,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"  Failed to create worktree: {e.stderr}")
            raise
        
        return self.worktree_dir
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up worktree, even if an exception occurred."""
        cleanup_worktree(self.worktree_dir)
        # Don't suppress exceptions
        return False


def cleanup_worktree(worktree_dir: Path) -> None:
    """Clean up a git worktree directory.
    
    Args:
        worktree_dir: Path to worktree directory
    """
    if not worktree_dir.exists():
        # Still prune in case git has a stale reference
        subprocess.run(
            ["git", "worktree", "prune"],
            capture_output=True,
            check=False,
        )
        return
        
    try:
        logging.debug(f"  Cleaning up worktree at {worktree_dir}")
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_dir)],
            capture_output=True,
            check=False,
        )
        # Force remove directory if git worktree remove failed
        if worktree_dir.exists():
            shutil.rmtree(worktree_dir, ignore_errors=True)
        
        # Prune stale worktree references from git's registry
        subprocess.run(
            ["git", "worktree", "prune"],
            capture_output=True,
            check=False,
        )
    except Exception as e:
        logging.debug(f"  Error during worktree cleanup: {e}")


def cleanup_all_stale_worktrees() -> None:
    """Clean up all stale experiment worktrees from previous runs.
    
    This is useful at script startup to clean up any worktrees left behind
    if the script was interrupted (e.g., with Ctrl-Z or Ctrl-C).
    """
    worktree_parent = Path("/tmp/ptyrax_gather_worktrees")
    if not worktree_parent.exists():
        worktree_parent.mkdir(parents=True, exist_ok=True)
        return
    
    # Find all worktree directories
    patterns = ["gather_worktree_*", "push_worktree_*"]
    stale_worktrees = []
    
    for pattern in patterns:
        stale_worktrees.extend(worktree_parent.glob(pattern))
    
    if stale_worktrees:
        logging.info(f"Cleaning up {len(stale_worktrees)} stale worktree(s) from previous runs...")
        for worktree_dir in stale_worktrees:
            cleanup_worktree(worktree_dir)
        # Final prune to ensure git's registry is clean
        subprocess.run(
            ["git", "worktree", "prune"],
            capture_output=True,
            check=False,
        )
        logging.info("Cleanup complete")


def run_dvc_command(cmd: List[str]) -> str:
    """Run a DVC command and return its output.
    
    Args:
        cmd: Command as list of strings
        
    Returns:
        Command output as string
    """
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout


def get_experiments_by_sweep_id(sweep_id: str) -> List[Dict[str, Any]]:
    """Get all experiments matching a sweep_id.
    
    Checks both applied experiments (via dvc exp show) and queued experiments
    (via git refs in .git/refs/exps/).
    
    Args:
        sweep_id: Sweep ID to filter by
        
    Returns:
        List of experiment dictionaries with commit, params, outputs
    """
    logging.info(f"Fetching experiments for sweep_id: {sweep_id}")
    
    experiments = []
    
    # Method 1: Get applied/committed experiments from dvc exp show
    try:
        output = run_dvc_command(["dvc", "exp", "show", "--json"])
        exp_data = json.loads(output)
        
        for branch_entry in exp_data:
            if "experiments" not in branch_entry or branch_entry["experiments"] is None:
                continue
                
            for exp_list in branch_entry["experiments"]:
                if "revs" not in exp_list or exp_list["revs"] is None:
                    continue
                    
                for exp in exp_list["revs"]:
                    exp_sweep_id = _extract_sweep_id_from_params(exp.get("data", {}).get("params", {}))
                    
                    if exp_sweep_id == sweep_id:
                        flat_params = _flatten_params(exp.get("data", {}).get("params", {}))
                        
                        experiments.append({
                            "name": exp.get("name", ""),
                            "commit": exp.get("rev", ""),
                            "params": flat_params,
                            "data": exp.get("data", {}),
                        })
        
        logging.info(f"Found {len(experiments)} applied experiments with sweep_id {sweep_id}")
    except Exception as e:
        logging.warning(f"Could not get experiments from dvc exp show: {e}")
    
    # Method 2: Get queued experiments from git refs (even if completed, they may not show in dvc exp show)
    try:
        queued_exps = _get_queued_experiments_from_refs(sweep_id)
        # Add queued experiments that aren't already in the list
        existing_names = {exp["name"] for exp in experiments}
        for exp in queued_exps:
            if exp["name"] not in existing_names:
                experiments.append(exp)
        
        logging.info(f"Found {len(queued_exps)} queued experiments, {len(experiments)} total")
    except Exception as e:
        logging.warning(f"Could not get queued experiments from git refs: {e}")
    
    # Method 3: Get experiments from remote branches (after push)
    try:
        remote_exps = _get_experiments_from_remote_branches(sweep_id)
        existing_names = {exp["name"] for exp in experiments}
        for exp in remote_exps:
            if exp["name"] not in existing_names:
                experiments.append(exp)
        
        logging.info(f"Found {len(remote_exps)} remote experiments, {len(experiments)} total")
    except Exception as e:
        logging.warning(f"Could not get experiments from remote branches: {e}")
    
    if not experiments:
        logging.warning(f"No experiments found for sweep_id {sweep_id}")
        logging.info("Note: Queued experiments must be explicitly applied/committed to show in 'dv exp show'")
        logging.info("The script now also checks .git/refs/exps/ and remote branches for experiments")
    
    return experiments


def _get_queued_experiments_from_refs(sweep_id: str) -> List[Dict[str, Any]]:
    """Get queued experiments by reading git refs directly.
    
    Queued experiments (even completed ones) don't show in 'dvc exp show' until
    they're explicitly applied and committed. This function reads them from
    git refs using git commands to catch all experiment ref formats.
    
    Args:
        sweep_id: Sweep ID to filter by
        
    Returns:
        List of experiment dictionaries with commit, params, outputs
    """
    experiments = []
    
    # Use git for-each-ref to find all experiment refs (handles nested structures)
    try:
        result = subprocess.run(
            ["git", "for-each-ref", "--format=%(refname)", "refs/exps/"],
            capture_output=True,
            text=True,
            check=True,
        )
        all_refs = result.stdout.strip().split('\n') if result.stdout.strip() else []
    except subprocess.CalledProcessError:
        return experiments
    
    # Special refs to skip (not real experiments)
    skip_refs = {'stash', 'failed', 'EXEC_APPLY', 'ORIG_HEAD', 'HEAD', 'FETCH_HEAD'}
    
    for ref in all_refs:
        # Extract experiment name from ref (last component)
        # e.g., refs/exps/a1/abb4.../exp-name -> exp-name
        exp_name = ref.split('/')[-1]
        
        # Skip special refs
        if exp_name in skip_refs or any(skip in ref for skip in skip_refs):
            continue
        
        try:
            # Get the commit hash for this experiment ref
            commit_hash = subprocess.run(
                ["git", "rev-parse", ref],
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            
            # Get parameters from the experiment's commit
            # Check dvc.lock which contains the resolved parameters
            try:
                dvc_lock_content = subprocess.run(
                    ["git", "show", f"{commit_hash}:dvc.lock"],
                    capture_output=True,
                    text=True,
                    check=True
                ).stdout
                
                dvc_lock = yaml.load(dvc_lock_content)
                
                # Extract sweep_id from stages
                exp_sweep_id = None
                for stage_name, stage_data in dvc_lock.get("stages", {}).items():
                    if "params" in stage_data:
                        exp_sweep_id = _extract_sweep_id_from_params(stage_data["params"])
                        if exp_sweep_id:
                            break
                
                if exp_sweep_id == sweep_id:
                    # Collect all params from all stages
                    all_params = {}
                    for stage_name, stage_data in dvc_lock.get("stages", {}).items():
                        if "params" in stage_data:
                            all_params.update(_flatten_params(stage_data["params"]))
                    
                    # Extract outputs from all stages
                    outs_dict = {}
                    for stage_name, stage_data in dvc_lock.get("stages", {}).items():
                        outs = stage_data.get("outs", [])
                        for out in outs:
                            if isinstance(out, dict) and "path" in out:
                                path = out["path"]
                                outs_dict[path] = out.get("md5", out.get("hash", ""))
                    
                    experiments.append({
                        "name": exp_name,
                        "commit": commit_hash,
                        "params": all_params,
                        "data": {
                            "params": dvc_lock.get("stages", {}),
                            "outs": outs_dict,
                        },
                    })
                    logging.debug(f"Found queued experiment: {exp_name} with sweep_id {exp_sweep_id}")
                    
            except subprocess.CalledProcessError:
                # This ref might not have a dvc.lock file, skip it
                logging.debug(f"Could not read dvc.lock for {exp_name}")
                continue
                
        except Exception as e:
            logging.debug(f"Could not process ref {exp_name}: {e}")
            continue
    
    return experiments


def _get_experiments_from_remote_branches(sweep_id: str, remote: str = "origin") -> List[Dict[str, Any]]:
    """Get experiments from remote branches matching sweep_id pattern.
    
    Looks for remote branches like {remote}/{sweep_id}/{exp_name}.
    
    Args:
        sweep_id: Sweep ID to match
        remote: Git remote name (default: "origin")
        
    Returns:
        List of experiment dictionaries
    """
    experiments = []
    
    # List remote branches
    try:
        result = subprocess.run(
            ["git", "branch", "-r"],
            capture_output=True,
            text=True,
            check=True,
        )
        branches = result.stdout.strip().split('\n')
        
        # Filter branches matching pattern: {remote}/{sweep_id}/*
        pattern_prefix = f"{remote}/{sweep_id}/"
        matching_branches = [b.strip() for b in branches if pattern_prefix in b]
        
        logging.debug(f"Found {len(matching_branches)} remote branches for sweep {sweep_id}")
        
        for branch in matching_branches:
            try:
                # Extract experiment name from branch (last part after last /)
                exp_name = branch.split('/')[-1]
                
                # Get commit hash for this branch
                commit_result = subprocess.run(
                    ["git", "rev-parse", branch],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                commit_hash = commit_result.stdout.strip()
                
                # Read dvc.lock from this commit
                lock_result = subprocess.run(
                    ["git", "show", f"{commit_hash}:dvc.lock"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                
                dvc_lock = yaml.load(lock_result.stdout)
                
                # Extract parameters from all stages - recursively flatten to find sweep_id
                all_params = {}
                for stage_name, stage_data in dvc_lock.get("stages", {}).items():
                    stage_params = stage_data.get("params", {})
                    if isinstance(stage_params, dict):
                        for file_key, file_params in stage_params.items():
                            if isinstance(file_params, dict):
                                # Recursively flatten to get all nested params
                                flattened = _flatten_params(file_params)
                                all_params.update(flattened)
                
                # Verify sweep_id matches
                exp_sweep_id = _extract_sweep_id_from_params(all_params)
                if exp_sweep_id != sweep_id:
                    logging.debug(f"Branch {branch} sweep_id mismatch: {exp_sweep_id} != {sweep_id}")
                    continue
                
                # Extract outputs from all stages
                outs_dict = {}
                for stage_name, stage_data in dvc_lock.get("stages", {}).items():
                    outs = stage_data.get("outs", [])
                    for out in outs:
                        if isinstance(out, dict) and "path" in out:
                            path = out["path"]
                            outs_dict[path] = out.get("md5", out.get("hash", ""))
                
                experiments.append({
                    "name": exp_name,
                    "commit": commit_hash,
                    "params": all_params,
                    "data": {
                        "params": dvc_lock.get("stages", {}),
                        "outs": outs_dict,
                    },
                })
                logging.debug(f"Found remote experiment: {exp_name} from branch {branch}")
                
            except subprocess.CalledProcessError as e:
                logging.debug(f"Could not process branch {branch}: {e}")
                continue
            except Exception as e:
                logging.debug(f"Error processing branch {branch}: {e}")
                continue
        
    except Exception as e:
        logging.warning(f"Could not list remote branches: {e}")
    
    return experiments


def _extract_sweep_id_from_params(params: Dict[str, Any]) -> Optional[str]:
    """Extract sweep_id from nested parameter dictionary.
    
    Args:
        params: Nested dictionary of parameters
        
    Returns:
        Sweep ID if found, None otherwise
    """
    # Check for direct sweep_id key
    if "sweep_id" in params and isinstance(params["sweep_id"], str):
        return params["sweep_id"]
    
    # Check for keys ending with sweep_id (for flattened dicts)
    for key, value in params.items():
        if key.endswith("sweep_id") and isinstance(value, str):
            return value
    
    # Recursively search nested dicts
    for key, value in params.items():
        if isinstance(value, dict):
            result = _extract_sweep_id_from_params(value)
            if result:
                return result
    
    return None


def _flatten_params(params: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested parameter dictionary.
    
    Args:
        params: Nested dictionary of parameters
        parent_key: Parent key for recursion
        sep: Separator for flattened keys
        
    Returns:
        Flattened dictionary with dot-separated keys
    """
    items = []
    for k, v in params.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            # Recursively flatten
            items.extend(_flatten_params(v, new_key, sep=sep).items())
        else:
            # Leaf value
            items.append((new_key, v))
    
    return dict(items)


def get_experiments_by_stage(stage_name: str) -> List[Dict[str, Any]]:
    """Get all experiments for a given stage.
    
    Args:
        stage_name: DVC stage name
        
    Returns:
        List of experiment dictionaries
    """
    logging.info(f"Fetching experiments for stage: {stage_name}")
    
    output = run_dvc_command(["dvc", "exp", "show", "--json"])
    exp_data = json.loads(output)
    
    experiments = []
    
    for branch_entry in exp_data:
        if "experiments" not in branch_entry or branch_entry["experiments"] is None:
            continue
            
        for exp_list in branch_entry["experiments"]:
            if "revs" not in exp_list or exp_list["revs"] is None:
                continue
                
            for exp in exp_list["revs"]:
                # Check if this experiment has outputs for the stage
                outs = exp.get("data", {}).get("outs", {})
                
                # Look for stage-specific outputs
                has_stage_output = any(stage_name in str(path) for path in outs.keys())
                
                if has_stage_output:
                    # Flatten nested params for easier access
                    flat_params = _flatten_params(exp.get("data", {}).get("params", {}))
                    
                    experiments.append({
                        "name": exp.get("name", ""),
                        "commit": exp.get("rev", ""),
                        "params": flat_params,
                        "data": exp.get("data", {}),
                    })
    
    logging.info(f"Found {len(experiments)} experiments for stage {stage_name}")
    return experiments


def parse_parameter_from_dirname(dirname: str) -> Dict[str, Any]:
    """Parse parameter values from log directory name.
    
    Args:
        dirname: Directory name with format YYYYMMDD_HHMMSS_exp-name_param1_value1_param2_value2
        
    Returns:
        Dictionary of parameter key-value pairs
    """
    params = {}
    
    # Split on underscore and look for param_value patterns
    parts = dirname.split("_")
    
    i = 3  # Skip timestamp and exp name parts
    while i < len(parts) - 1:
        param_name = parts[i]
        param_value = parts[i + 1]
        
        # Try to convert to float if possible
        try:
            param_value = float(param_value)
        except ValueError:
            pass
        
        params[param_name] = param_value
        i += 2
    
    return params


def find_log_directories(base_dir: Path, exp_name: str, stage_name: Optional[str] = None) -> List[Path]:
    """Find log directories for an experiment.
    
    Args:
        base_dir: Base directory to search (e.g., reconstructions/)
        exp_name: Experiment name
        stage_name: Optional stage name to filter by
        
    Returns:
        List of log directory paths
    """
    log_dirs = []
    
    # Search in stage-specific subdirectory if provided
    if stage_name:
        search_dir = base_dir / stage_name
    else:
        search_dir = base_dir
    
    if not search_dir.exists():
        return log_dirs
    
    # Look for directories containing the experiment name
    for subdir in search_dir.iterdir():
        if subdir.is_dir() and exp_name in subdir.name:
            log_dirs.append(subdir)
    
    return log_dirs


def load_tensorboard_hdf5(tb_hdf5_path: Path) -> Dict[str, Any]:
    """Load TensorBoard logs from HDF5 file.
    
    Args:
        tb_hdf5_path: Path to tensorboard_logs.hdf5
        
    Returns:
        Dictionary with training data (scalars, histograms, etc.)
    """
    data = {}
    
    with h5py.File(tb_hdf5_path, 'r') as f:
        def copy_group(h5_group, target_dict):
            for key in h5_group.keys():
                if isinstance(h5_group[key], h5py.Group):
                    target_dict[key] = {}
                    copy_group(h5_group[key], target_dict[key])
                else:
                    target_dict[key] = h5_group[key][()]
        
        copy_group(f, data)
    
    return data


def _find_modified_hdf5_in_folder(folder: Path, commit_sha: str) -> List[Path]:
    """Find HDF5 and CXI files in folder that were modified vs git commit.
    
    Args:
        folder: Folder to search
        commit_sha: Git commit SHA to compare against
        
    Returns:
        List of HDF5/CXI file paths that are new or modified
    """
    hdf5_files = []
    
    if not folder.exists() or not folder.is_dir():
        return hdf5_files
    
    # Get all HDF5 and CXI files in folder (recursively)
    # CXI is Coherent X-ray Imaging format, which is HDF5-based
    for pattern in ["*.hdf5", "*.cxi"]:
        for hdf5_file in folder.rglob(pattern):
            if hdf5_file.name == "tensorboard_logs.hdf5":
                continue
            
            # Check if file exists in git at this commit
            try:
                # Get relative path from repo root
                repo_root = subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    capture_output=True,
                    text=True,
                    check=True
                ).stdout.strip()
                
                rel_path = hdf5_file.relative_to(repo_root)
                
                # Try to get file from git at this commit
                result = subprocess.run(
                    ["git", "show", f"{commit_sha}:{rel_path}"],
                    capture_output=True,
                    check=False
                )
                
                # If file doesn't exist at commit or is different, include it
                if result.returncode != 0:
                    hdf5_files.append(hdf5_file)
                    
            except (subprocess.CalledProcessError, ValueError):
                # If we can't check git, include the file
                hdf5_files.append(hdf5_file)
    
    return hdf5_files


def get_dvc_stage_outputs(stage_name: str) -> List[Path]:
    """Get output paths for a DVC stage.
    
    Args:
        stage_name: Name of DVC stage
        
    Returns:
        List of output paths
    """
    dvc_yaml_path = Path("dvc.yaml")
    
    if not dvc_yaml_path.exists():
        return []
    
    with open(dvc_yaml_path, 'r') as f:
        dvc_config = yaml.load(f)
    
    stages = dvc_config.get("stages", {})
    stage_config = stages.get(stage_name, {})
    
    outputs = []
    for out in stage_config.get("outs", []):
        if isinstance(out, str):
            outputs.append(Path(out))
        elif isinstance(out, dict):
            for key in out.keys():
                outputs.append(Path(key))
    
    return outputs


def extract_sweep_id_from_tensorboard(tb_hdf5_path: Path) -> Optional[str]:
    """Extract sweep_id from TensorBoard HDF5 file.
    
    Args:
        tb_hdf5_path: Path to tensorboard_logs.hdf5
        
    Returns:
        Sweep ID if found, None otherwise
    """
    try:
        with h5py.File(tb_hdf5_path, 'r') as f:
            # Check for sweep_id in text data
            if 'scalars/sweep_id/value' in f:
                # Read the binary data
                binary_data = f['scalars/sweep_id/value'][0]
                
                # Extract text from protocol buffer format
                # Format is typically: b'\x08\x07\x12\x04\x12\x02\x08\x01B\x1a{text}'
                if isinstance(binary_data, bytes):
                    text = binary_data.decode('utf-8', errors='ignore')
                else:
                    text = str(binary_data)
                
                # Use regex to extract sweep_id pattern
                match = re.search(r'\d{8}_\d{6}_[a-z]+-[a-z]+', text)
                if match:
                    return match.group(0)
    except Exception as e:
        logging.debug(f"Could not extract sweep_id from {tb_hdf5_path}: {e}")
    
    return None


def copy_hdf5_tree(src_file: Path, dst_group: h5py.Group, group_name: str):
    """Copy entire HDF5 file tree into a group.
    
    Args:
        src_file: Source HDF5 file
        dst_group: Destination HDF5 group
        group_name: Name for the new group
    """
    with h5py.File(src_file, 'r') as src:
        # Create target group
        target = dst_group.create_group(group_name)
        
        # Recursively copy structure
        def copy_item(name, obj):
            if isinstance(obj, h5py.Group):
                target.create_group(name)
                for key in obj.keys():
                    copy_item(f"{name}/{key}", obj[key])
            else:
                # Dataset
                target[name] = obj[()]
                # Copy attributes
                for attr_name, attr_value in obj.attrs.items():
                    target[name].attrs[attr_name] = attr_value
        
        # Copy all items from root
        for key in src.keys():
            copy_item(key, src[key])


def find_dependency_stages(stage_name: str) -> List[str]:
    """Find stages that the given stage depends on.
    
    Args:
        stage_name: Name of DVC stage
        
    Returns:
        List of dependency stage names
    """
    dvc_yaml_path = Path("dvc.yaml")
    
    if not dvc_yaml_path.exists():
        return []
    
    with open(dvc_yaml_path, 'r') as f:
        dvc_config = yaml.load(f)
    
    stages = dvc_config.get("stages", {})
    stage_config = stages.get(stage_name, {})
    
    # Get dependencies
    deps = stage_config.get("deps", [])
    
    # Find which stages produce these dependencies
    dep_stages = []
    
    for dep in deps:
        if isinstance(dep, dict):
            dep_path = list(dep.keys())[0]
        else:
            dep_path = dep
        
        # Check which stages output this path
        for other_stage_name, other_stage_config in stages.items():
            if other_stage_name == stage_name:
                continue
            
            other_outs = other_stage_config.get("outs", [])
            for out in other_outs:
                if isinstance(out, dict):
                    out_path = list(out.keys())[0]
                else:
                    out_path = out
                
                # Check if dependency is under this output
                if dep_path.startswith(out_path):
                    if other_stage_name not in dep_stages:
                        dep_stages.append(other_stage_name)
    
    return dep_stages


def gather_experiment_metadata(
    experiment: Dict[str, Any],
    log_base_dirs: List[Path],
    target_sweep_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Gather only metadata for an experiment (no worktree creation).
    
    This is used for the first pass to build HDF5 index without consuming disk space.
    
    Args:
        experiment: Experiment dictionary from DVC
        log_base_dirs: Base directories to search for logs
        target_sweep_id: Optional sweep ID to verify
        
    Returns:
        Dictionary with experiment metadata (name, params, log_dir, stage_name)
    """
    exp_name = experiment["name"]
    
    logging.debug(f"Gathering metadata for experiment: {exp_name}")
    
    # Determine ALL stages from experiment outputs (don't just pick one!)
    stage_names = set()
    outs = experiment.get("data", {}).get("outs", {})
    for out_path in outs.keys():
        parts = Path(out_path).parts
        if len(parts) >= 2:
            potential_stage = parts[1]
            if get_dvc_stage_outputs(potential_stage):
                stage_names.add(potential_stage)
    
    # Use first stage for log directory finding (but we'll track all stages)
    stage_name = list(stage_names)[0] if stage_names else None
    
    # Find log directory
    log_dir = None
    for base_dir in log_base_dirs:
        log_dirs = find_log_directories(base_dir, exp_name, stage_name)
        if log_dirs:
            log_dir = log_dirs[0]
            break
    
    if not log_dir:
        logging.warning(f"No log directory found for {exp_name}")
        return None
    
    # Get params from experiment data (DVC already has them)
    params = experiment.get("params", {})
    
    # Check sweep_id if provided (from tensorboard logs)
    if target_sweep_id:
        training_data_file = Path(log_dir) / "tensorboard_logs.hdf5"
        if training_data_file.exists():
            try:
                with h5py.File(training_data_file, "r") as f:
                    file_sweep_id = f.attrs.get("sweep_id")
                    if file_sweep_id != target_sweep_id:
                        logging.warning(f"Sweep ID mismatch: expected {target_sweep_id}, found {file_sweep_id}")
            except Exception as e:
                logging.debug(f"Could not check sweep_id: {e}")
    
    # Get all stages: ALL detected stages + their dependencies
    all_stages_available = []
    for s in sorted(stage_names):  # Process in consistent order
        if s not in all_stages_available:
            all_stages_available.append(s)
        # Add dependencies of each stage
        deps = find_dependency_stages(s)
        for d in deps:
            if d not in all_stages_available:
                all_stages_available.append(d)
    
    return {
        "name": exp_name,
        "params": params,
        "log_dir": log_dir,
        "stage_name": stage_name,
        "all_stages": all_stages_available,
    }


def gather_experiment_data(
    experiment: Dict[str, Any],
    log_base_dirs: List[Path],
    target_sweep_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Gather all data for a single experiment.
    
    Args:
        experiment: Experiment dictionary from DVC
        log_base_dirs: Base directories to search for logs
        target_sweep_id: Optional sweep ID to verify
        
    Returns:
        Dictionary with experiment data
    """
    exp_name = experiment["name"]
    exp_commit = experiment.get("commit", "HEAD")
    
    logging.info(f"Gathering data for experiment: {exp_name}")
    
    # Determine all stages from experiment outputs
    stage_names = set()
    outs = experiment.get("data", {}).get("outs", {})
    for out_path in outs.keys():
        # Extract stage name from path (e.g., data/SI0009_reconstruct -> SI0009_reconstruct)
        parts = Path(out_path).parts
        if len(parts) >= 2:
            potential_stage = parts[1]
            # Verify it's a real stage
            if get_dvc_stage_outputs(potential_stage):
                stage_names.add(potential_stage)
    
    # Use the first stage for log directory detection
    stage_name = list(stage_names)[0] if stage_names else None
    all_stages = list(stage_names)  # Keep ALL detected stages
    
    if stage_name:
        logging.info(f"Sweep corresponds to stages: {all_stages}")
    
    # Find log directory
    log_dir = None
    for base_dir in log_base_dirs:
        log_dirs = find_log_directories(base_dir, exp_name, stage_name)
        if log_dirs:
            log_dir = log_dirs[0]
            logging.info(f"Found log directory: {log_dir}")
            break
    
    if not log_dir:
        logging.warning(f"No log directory found for {exp_name}")
    
    # Find output files from all stages
    # IMPORTANT: Must create a worktree and gather from there to get experiment-specific outputs
    # (not from main workspace which may have different experiment applied)
    output_files = {}
    tensorboard_file = None  # Will be set when we find it
    worktree_dir = None
    
    # Also gather outputs from dependent stages (e.g., SI0009_simulate when running SI0009_reconstruct)
    dependent_stages = find_dependency_stages(stage_name) if stage_name else []
    
    # Always create worktree to get correct experiment outputs
    # Note: We can't use context manager here because worktree must stay alive
    # until files are copied to HDF5 in create_pooled_hdf5()
    exp_commit = experiment.get("commit", "HEAD")
    worktree_dir = Path(f"/tmp/ptyrax_gather_worktrees/gather_worktree_{exp_name}_{exp_commit[:8]}")
    
    try:
        # Clean up any existing worktree at this location
        cleanup_worktree(worktree_dir)
        
        # Create git worktree at the experiment's original commit
        logging.debug(f"    Creating worktree at {worktree_dir} from commit {exp_commit[:8]}")
        try:
            result = subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree_dir), exp_commit],
                capture_output=True,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logging.warning(f"    Failed to create worktree: {e.stderr.strip()}")
            raise
        
        # Apply the experiment state using dvc exp apply
        logging.debug(f"    Applying experiment {exp_name} in worktree")
        result = subprocess.run(
            ["dvc", "exp", "apply", exp_name],
            cwd=worktree_dir,
            capture_output=True,
        )
        
        if result.returncode != 0:
            logging.warning(f"    Failed to apply experiment {exp_name}: {result.stderr.decode()[:200]}")
        else:
            # Checkout DVC-tracked files to materialize outputs from cache
            logging.debug(f"    Checking out DVC outputs in worktree")
            checkout_result = subprocess.run(
                ["dvc", "checkout"],
                cwd=worktree_dir,
                capture_output=True,
            )
            
            if checkout_result.returncode != 0:
                logging.warning(f"    DVC checkout warning: {checkout_result.stderr.decode()[:200]}")
            
            # Now gather outputs from ALL stages (including main and dependent)
            # We need to copy the files IMMEDIATELY while the worktree exists
            all_stages_to_gather = all_stages + [s for s in dependent_stages if s not in all_stages]
            
            for current_stage in all_stages_to_gather:
                is_dependent = current_stage in dependent_stages
                stage_label = "dependent stage" if is_dependent else "stage"
                logging.info(f"  Gathering outputs from {stage_label}: {current_stage}")
                stage_outputs = get_dvc_stage_outputs(current_stage)
                
                for stage_output in stage_outputs:
                    stage_output_in_worktree = worktree_dir / stage_output
                    
                    # Gather file paths from the worktree (don't copy to temp)
                    if stage_output_in_worktree.exists():
                        if stage_output_in_worktree.is_file() and stage_output_in_worktree.suffix in [".hdf5", ".cxi"]:
                            # Prefix key with stage name if multiple stages
                            key = stage_output_in_worktree.stem
                            if len(all_stages_to_gather) > 1:
                                key = f"{current_stage}/{key}"
                            # Store the path in the worktree (no temp file)
                            output_files[key] = stage_output_in_worktree
                            logging.debug(f"      Added {key} (in worktree)")
                        elif stage_output_in_worktree.is_dir():
                            # Directory output - search for HDF5/CXI files
                            for pattern in ["*.hdf5", "*.cxi"]:
                                for hdf5_file in stage_output_in_worktree.rglob(pattern):
                                    # Special handling for tensorboard_logs.hdf5
                                    if hdf5_file.name == "tensorboard_logs.hdf5":
                                        if not tensorboard_file:  # Use first one found
                                            tensorboard_file = hdf5_file
                                            logging.info(f"      Found tensorboard_logs.hdf5")
                                        continue
                                    try:
                                        rel_path = hdf5_file.relative_to(stage_output_in_worktree)
                                        key = str(rel_path.with_suffix('')).replace('/', '_')
                                        if len(all_stages_to_gather) > 1:
                                            key = f"{current_stage}/{key}"
                                    except ValueError:
                                        key = hdf5_file.stem
                                        if len(all_stages_to_gather) > 1:
                                            key = f"{current_stage}/{key}"
                                    if key not in output_files:
                                        # Store the path in the worktree (no temp file)
                                        output_files[key] = hdf5_file
                                        logging.debug(f"      Added {key} (in worktree)")
    
    except Exception as e:
        logging.warning(f"  Could not create worktree for dependent stages: {e}")
        # Clean up on error
        if worktree_dir:
            cleanup_worktree(worktree_dir)
    
    # NOTE: We DON'T clean up the worktree here - it stays alive
    # It will be cleaned up in create_pooled_hdf5() after copying to HDF5
    # This avoids filling up /tmp with copies of large files
    
    # Also check in log directory for additional files (but NOT recursively in parent dirs)
    # Only look in the specific experiment's log directory to avoid picking up old experiments
    search_dirs = [log_dir] if log_dir else []  # Only search the experiment's own log directory, not parents
    
    for search_dir in search_dirs:
        # Check for direct HDF5 and CXI files (non-recursive)
        for pattern in ["*.hdf5", "*.cxi"]:
            for hdf5_file in search_dir.glob(pattern):
                if hdf5_file.name != "tensorboard_logs.hdf5":
                    # Use stem as key (filename without extension)
                    key = hdf5_file.stem
                    if key not in output_files:
                        output_files[key] = hdf5_file
    
    # Load training data from tensorboard_logs.hdf5 (prefer output dir, fallback to log_dir)
    training_data = {}
    if tensorboard_file and tensorboard_file.exists():
        logging.info(f"Loading training data from {tensorboard_file}")
        training_data = load_tensorboard_hdf5(tensorboard_file)
        if target_sweep_id:
            found_sweep_id = extract_sweep_id_from_tensorboard(tensorboard_file)
            if found_sweep_id != target_sweep_id:
                logging.warning(f"Sweep ID mismatch: expected {target_sweep_id}, found {found_sweep_id}")
    elif log_dir:
        # Fallback: try log_dir if tensorboard file wasn't in outputs
        tb_hdf5 = log_dir / "tensorboard_logs.hdf5"
        if tb_hdf5.exists():
            logging.info(f"Loading training data from log_dir: {tb_hdf5}")
            training_data = load_tensorboard_hdf5(tb_hdf5)
            if target_sweep_id:
                found_sweep_id = extract_sweep_id_from_tensorboard(tb_hdf5)
                if found_sweep_id != target_sweep_id:
                    logging.warning(f"Sweep ID mismatch: expected {target_sweep_id}, found {found_sweep_id}")
    
    return {
        "name": exp_name,
        "log_dir": log_dir,
        "params": experiment.get("params", {}),
        "training_data": training_data,
        "output_files": output_files,
        "worktree_dir": worktree_dir,  # Keep worktree alive for direct copying to HDF5
    }


def are_ground_truths_equal(file1: Path, file2: Path) -> Tuple[bool, Optional[str]]:
    """Check if ground truth data is identical between two HDF5 files.
    
    Args:
        file1: First HDF5 file  
        file2: Second HDF5 file
        
    Returns:
        Tuple of (is_equal, ground_truth_path) where ground_truth_path is the path to sample data
    """
    try:
        with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
            # Find sample/ground truth groups in both files
            sample_paths_1 = []
            sample_paths_2 = []
            
            def find_sample_groups(group, path="", results=[]):
                for key in group.keys():
                    current_path = f"{path}/{key}" if path else key
                    if 'sample' in key.lower() or 'ground_truth' in key.lower():
                        results.append(current_path)
                    if isinstance(group[key], h5py.Group):
                        find_sample_groups(group[key], current_path, results)
            
            find_sample_groups(f1, results=sample_paths_1)
            find_sample_groups(f2, results=sample_paths_2)
            
            if not sample_paths_1 or not sample_paths_2:
                return (False, None)
            
            # Compare the first matching sample group
            gt_path = sample_paths_1[0]
            if gt_path not in f2:
                return (False, None)
            
            # Compare datasets recursively (except thickness)
            def compare_groups(g1, g2):
                keys1 = set(g1.keys())
                keys2 = set(g2.keys())
                
                # Allow for thickness to be different
                keys1_filtered = {k for k in keys1 if 'thickness' not in k.lower()}
                keys2_filtered = {k for k in keys2 if 'thickness' not in k.lower()}
                
                if keys1_filtered != keys2_filtered:
                    return False
                
                for key in keys1_filtered:
                    obj1 = g1[key]
                    obj2 = g2[key]
                    
                    if isinstance(obj1, h5py.Group) and isinstance(obj2, h5py.Group):
                        if not compare_groups(obj1, obj2):
                            return False
                    elif isinstance(obj1, h5py.Dataset) and isinstance(obj2, h5py.Dataset):
                        if not np.array_equal(obj1[()], obj2[()], equal_nan=True):
                            return False
                    else:
                        return False
                
                return True
            
            is_equal = compare_groups(f1[gt_path], f2[gt_path])
            return (is_equal, gt_path if is_equal else None)
            
    except Exception as e:
        logging.debug(f"Error comparing ground truths: {e}")
        return (False, None)


def apply_commit_push_experiments(
    experiments: List[Dict[str, Any]],
    sweep_id: str,
    git_remote: str = "origin",
    dvc_remote: Optional[str] = None,
) -> None:
    """Apply, commit, and push all experiments in a sweep to git/DVC remotes.
    
    This function iteratively:
    1. Creates a worktree at the experiment's parent commit
    2. Applies the experiment using `dvc exp apply`
    3. Commits the changes
    4. Pushes to git and DVC remotes
    5. Cleans up the worktree
    
    Args:
        experiments: List of experiment dictionaries (from get_experiments_by_sweep_id)
        sweep_id: Sweep ID for logging
        git_remote: Git remote name (default: "origin")
        dvc_remote: DVC remote name. If None, DVC uses its default remote.
    """
    logging.info(f"Applying, committing, and pushing {len(experiments)} experiments from sweep {sweep_id}")
    
    successful_pushes = []
    failed_pushes = []
    
    for i, exp in enumerate(experiments, 1):
        exp_name = exp.get("name", "unknown")
        exp_commit = exp.get("commit", "")
        
        logging.info(f"[{i}/{len(experiments)}] Processing experiment: {exp_name}")
        
        try:
            # Use context manager to ensure worktree cleanup even if interrupted
            with GitWorktree(exp_name, exp_commit, prefix="push") as worktree_dir:
                # Apply the experiment in the worktree
                logging.info(f"  Applying experiment {exp_name}")
                subprocess.run(
                    ["dvc", "exp", "apply", exp_name],
                    cwd=worktree_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                
                # Commit DVC outputs (updates .dvc files and dvc.lock)
                logging.info(f"  Committing DVC outputs")
                result = subprocess.run(
                    ["dvc", "commit", "--force"],
                    cwd=worktree_dir,
                    capture_output=True,
                    text=True,
                    check=False,  # Don't fail if no changes
                )
                if result.returncode != 0:
                    logging.debug(f"    dvc commit output: {result.stdout}")
                    logging.debug(f"    dvc commit stderr: {result.stderr}")
                
                # Stage DVC-related files
                logging.info(f"  Staging changes")
                subprocess.run(
                    ["git", "add", "*.dvc", "dvc.lock", ".gitignore"],
                    cwd=worktree_dir,
                    capture_output=True,
                    text=True,
                    check=False,  # Don't fail if some files don't exist
                )
                
                # Check if there are actually changes to commit
                result = subprocess.run(
                    ["git", "diff", "--cached", "--quiet"],
                    cwd=worktree_dir,
                    capture_output=True,
                    check=False,
                )
                has_changes = result.returncode != 0
                
                if not has_changes:
                    logging.info(f"  No changes to commit for {exp_name}, skipping")
                    successful_pushes.append(exp_name)
                    continue
                
                # Commit the changes
                commit_msg = f"Apply experiment {exp_name} from sweep {sweep_id}"
                logging.info(f"  Committing git changes")
                subprocess.run(
                    ["git", "commit", "-m", commit_msg],
                    cwd=worktree_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                
                # Get the new commit hash
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=worktree_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                new_commit = result.stdout.strip()
                
                # Push to git remote
                logging.info(f"  Pushing to git remote {git_remote}")
                subprocess.run(
                    ["git", "push", git_remote, f"{new_commit}:refs/heads/{sweep_id}/{exp_name}"],
                    cwd=worktree_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                
                # Push DVC data
                if dvc_remote:
                    logging.info(f"  Pushing DVC data to remote {dvc_remote}")
                    dvc_cmd = ["dvc", "push", "-r", dvc_remote]
                else:
                    logging.info(f"  Pushing DVC data to default remote")
                    dvc_cmd = ["dvc", "push"]
                
                subprocess.run(
                    dvc_cmd,
                    cwd=worktree_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                
                logging.info(f"  ✓ Successfully pushed experiment {exp_name}")
                successful_pushes.append(exp_name)
            
        except subprocess.CalledProcessError as e:
            logging.error(f"  ✗ Failed to push experiment {exp_name}: {e}")
            logging.debug(f"    stdout: {e.stdout}")
            logging.debug(f"    stderr: {e.stderr}")
            failed_pushes.append(exp_name)
            
        except Exception as e:
            logging.error(f"  ✗ Failed to push experiment {exp_name}: {e}")
            failed_pushes.append(exp_name)
    
    # Summary
    logging.info(f"\nPush summary:")
    logging.info(f"  Successful: {len(successful_pushes)}/{len(experiments)}")
    if failed_pushes:
        logging.warning(f"  Failed: {len(failed_pushes)}/{len(experiments)}")
        logging.warning(f"  Failed experiments: {', '.join(failed_pushes)}")


def create_pooled_hdf5_streaming(
    experiments: List[Dict[str, Any]],
    experiments_metadata: List[Dict[str, Any]],
    output_path: Path,
    sweep_id: str,
    log_base_dirs: List[Path],
    include_stages: Optional[List[str]] = None,
    exclude_stages: Optional[List[str]] = None,
):
    """Create pooled HDF5 file with streaming worktree creation.
    
    Creates worktrees one at a time, copies files, then cleans up immediately
    to minimize disk space usage.
    
    Args:
        experiments: List of raw experiment dicts from DVC
        experiments_metadata: List of metadata-only dicts (no worktrees)
        output_path: Output HDF5 file path
        sweep_id: Sweep ID for metadata
        log_base_dirs: Base directories for log files
        include_stages: If provided, only include these stages
        exclude_stages: If provided, exclude these stages
    """
    logging.info(f"Creating pooled HDF5 with streaming: {output_path}")
    
    with h5py.File(output_path, "w") as f:
        # Store sweep metadata
        f.attrs["sweep_id"] = sweep_id
        f.attrs["n_experiments"] = len(experiments_metadata)
        
        # Create index table
        index_group = f.create_group("index")
        
        # Build index arrays from metadata (no worktrees needed)
        exp_names = []
        param_keys = set()
        for exp_meta in experiments_metadata:
            exp_names.append(exp_meta["name"])
            param_keys.update(exp_meta["params"].keys())
        
        param_keys = sorted(param_keys)
        
        # Store index data
        index_group.create_dataset(
            "exp_names",
            data=np.array(exp_names, dtype=h5py.string_dtype()),
        )
        index_group.create_dataset(
            "indices",
            data=np.arange(len(exp_names)),
        )
        
        # Store parameter columns
        seen_keys = set()
        for param_key in param_keys:
            param_values = []
            for exp_meta in experiments_metadata:
                value = exp_meta["params"].get(param_key, np.nan)
                param_values.append(value)
            
            # Clean up parameter name - extract the actual parameter name
            clean_key = param_key.split('.')[-1]  # Remove path prefixes
            for param_name in ["thickness", "amplitude", "phase", "rotation", "scale"]:
                if param_key.endswith(param_name):
                    clean_key = param_name
                    break
            
            # Handle duplicate keys
            if clean_key in seen_keys:
                clean_key = param_key.replace('.', '_').replace('/', '_')
            seen_keys.add(clean_key)
            
            # Try to convert to numeric array, fall back to strings if needed
            try:
                param_array = np.array(param_values, dtype=float)
            except (ValueError, TypeError):
                # If values can't be converted to float, store as strings
                param_array = np.array([str(v) for v in param_values], dtype=h5py.string_dtype())
            
            index_group.create_dataset(
                f"param_{clean_key}",
                data=param_array,
            )
        
        # Store each experiment's data (streaming worktree creation)
        first_exp_output_files = {}  # Track first experiment's outputs for deduplication
        
        for i, (exp, exp_meta) in enumerate(zip(experiments, experiments_metadata)):
            exp_name = exp_meta["name"]
            logging.info(f"Adding experiment {i+1}/{len(experiments_metadata)}: {exp_name}")
            
            # Create worktree for this experiment only
            exp_commit = exp.get("commit", "HEAD")
            worktree_dir = Path(f"/tmp/ptyrax_gather_worktrees/gather_worktree_{exp_name}_{exp_commit[:8]}")
            
            try:
                # Clean up any existing worktree
                cleanup_worktree(worktree_dir)
                
                # Create git worktree
                logging.debug(f"  Creating worktree at {worktree_dir} from commit {exp_commit[:8]}")
                subprocess.run(
                    ["git", "worktree", "add", "--detach", str(worktree_dir), exp_commit],
                    capture_output=True,
                    check=True,
                    text=True,
                )
                
                # Apply experiment state
                logging.debug(f"  Applying experiment {exp_name} in worktree")
                result = subprocess.run(
                    ["dvc", "exp", "apply", exp_name],
                    cwd=worktree_dir,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    logging.warning(f"  DVC exp apply had issues: {result.stderr.strip()[:200]}")
                
                # DVC checkout to get outputs
                logging.debug(f"  Running DVC checkout to get stage outputs")
                result = subprocess.run(
                    ["dvc", "checkout"],
                    cwd=worktree_dir,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    logging.info(f"  ⚠ DVC checkout failed - outputs may not be available in cache")
                    if "missing files" in result.stderr.lower() or "not in cache" in result.stderr.lower():
                        logging.info(f"  💡 Tip: Stage outputs aren't in DVC cache. Training data will still be gathered from log_dir.")
                    logging.debug(f"  DVC checkout error: {result.stderr.strip()[:400]}")
                
            except Exception as e:
                logging.error(f"  Failed to create worktree for {exp_name}: {e}")
                cleanup_worktree(worktree_dir)
                continue
            
            exp_group = f.create_group(exp_name)
            
            # Store metadata
            meta_group = exp_group.create_group("metadata")
            
            # Store log_dir as dataset
            meta_group.create_dataset("log_dir", data=str(exp_meta["log_dir"]))
            
            # Store each parameter as a dataset
            seen_meta_keys = set(["log_dir"])
            for key, value in exp_meta["params"].items():
                # Skip dict values - they shouldn't be here after flattening
                if isinstance(value, dict):
                    logging.warning(f"Skipping nested dict parameter: {key}")
                    continue
                
                # Clean up parameter names - extract the actual parameter name
                # e.g., "symmetric_trans_newcoordsthickness" -> "thickness"
                clean_key = key.split('.')[-1]  # Remove path prefixes
                # If key ends with a parameter name, extract it
                for param_name in ["thickness", "amplitude", "phase", "rotation", "scale"]:
                    if key.endswith(param_name):
                        clean_key = param_name
                        break
                
                # Handle duplicate keys
                if clean_key in seen_meta_keys:
                    clean_key = key.replace('.', '_').replace('/', '_')
                seen_meta_keys.add(clean_key)
                
                # Store as dataset with proper type handling
                try:
                    if isinstance(value, (list, tuple, np.ndarray)):
                        # Try to convert to array
                        try:
                            arr = np.array(value)
                            # If array dtype is object, convert to string
                            if arr.dtype == object:
                                meta_group.create_dataset(clean_key, data=str(value), dtype=h5py.string_dtype())
                            else:
                                meta_group.create_dataset(clean_key, data=arr)
                        except (TypeError, ValueError):
                            # Fall back to string
                            meta_group.create_dataset(clean_key, data=str(value), dtype=h5py.string_dtype())
                    elif isinstance(value, (int, float, np.number)):
                        meta_group.create_dataset(clean_key, data=value)
                    elif isinstance(value, bool):
                        meta_group.create_dataset(clean_key, data=int(value))
                    else:
                        meta_group.create_dataset(clean_key, data=str(value), dtype=h5py.string_dtype())
                except (TypeError, ValueError) as e:
                    # Fall back to string if type conversion fails
                    meta_group.create_dataset(clean_key, data=str(value), dtype=h5py.string_dtype())
            
            # Gather training data and output files from worktree
            try:
                # Load training data from log_dir
                log_dir = exp_meta["log_dir"]
                training_data_file = Path(log_dir) / "tensorboard_logs.hdf5"
                
                # Store training data by copying entire HDF5 structure
                if training_data_file.exists():
                    logging.info(f"  Loading training data from {training_data_file}")
                    try:
                        # Copy the entire HDF5 tree structure, not just attributes
                        copy_hdf5_tree(training_data_file, exp_group, "training")
                    except Exception as e:
                        logging.warning(f"  Could not load training data: {e}")
                        # Create empty training group as fallback
                        exp_group.create_group("training")
                else:
                    # Create empty training group if no data file
                    exp_group.create_group("training")
                
                # Find output files from worktree using stage outputs
                all_stages_available = exp_meta.get("all_stages", [])
                
                # Apply stage filters
                all_stages_to_gather = filter_stages(
                    all_stages_available,
                    include_stages=include_stages,
                    exclude_stages=exclude_stages,
                )
                
                if not all_stages_to_gather:
                    logging.warning(f"  No stages to gather for {exp_name} after filtering")
                
                output_files = {}
                for current_stage in all_stages_to_gather:
                    logging.info(f"  Gathering outputs from stage: {current_stage}")
                    stage_outputs = get_dvc_stage_outputs(current_stage)
                    
                    for stage_output in stage_outputs:
                        stage_output_in_worktree = worktree_dir / stage_output
                        
                        # Check if output exists in worktree
                        if stage_output_in_worktree.exists():
                            if stage_output_in_worktree.is_file() and stage_output_in_worktree.suffix in [".hdf5", ".cxi"]:
                                # Prefix key with stage name if multiple stages
                                key = stage_output_in_worktree.stem
                                if len(all_stages_to_gather) > 1:
                                    key = f"{current_stage}/{key}"
                                output_files[key] = stage_output_in_worktree
                                logging.debug(f"      Found {key}")
                            elif stage_output_in_worktree.is_dir():
                                # Directory output - search for HDF5/CXI files
                                for pattern in ["*.hdf5", "*.cxi"]:
                                    for hdf5_file in stage_output_in_worktree.rglob(pattern):
                                        # Skip tensorboard_logs.hdf5 (already handled above)
                                        if hdf5_file.name == "tensorboard_logs.hdf5":
                                            continue
                                        try:
                                            rel_path = hdf5_file.relative_to(stage_output_in_worktree)
                                            key = str(rel_path.with_suffix('')).replace('/', '_')
                                            if len(all_stages_to_gather) > 1:
                                                key = f"{current_stage}/{key}"
                                        except ValueError:
                                            key = hdf5_file.stem
                                            if len(all_stages_to_gather) > 1:
                                                key = f"{current_stage}/{key}"
                                        if key not in output_files:
                                            output_files[key] = hdf5_file
                                            logging.debug(f"      Found {key}")
                
                logging.info(f"  Found {len(output_files)} output file(s) to copy")
                
                # Store output files with optimized ground truth handling
                for output_name, output_file in output_files.items():
                    logging.info(f"  Copying {output_name} from {output_file}")
                    try:
                        # Check if this contains ground truth data (simulation output)
                        is_simulation = "simulate" in output_name.lower()
                        
                        if is_simulation and i > 0 and first_exp_output_files:
                            # Check if ground truth is same as first experiment
                            first_output_file = first_exp_output_files.get(output_name)
                            
                            if first_output_file:
                                is_equal, gt_path = are_ground_truths_equal(output_file, first_output_file)
                                
                                if is_equal and gt_path:
                                    logging.info(f"    Ground truth identical to first experiment - creating reference")
                                    
                                    # Store reference to first experiment's ground truth
                                    first_exp_name = experiments_metadata[0]["name"]
                                    exp_group.attrs[f"{output_name}_ground_truth_ref"] = first_exp_name
                                    
                                    # Only copy thickness values (or other varying parameters)
                                    with h5py.File(output_file, 'r') as src:
                                        if gt_path in src:
                                            # Create minimal copy with only varying params
                                            output_subgroup = exp_group.create_group(output_name)
                                            gt_grp = src[gt_path]
                                            params_grp = output_subgroup.create_group(gt_path)
                                            
                                            # Copy only varying parameters
                                            for key in gt_grp.keys():
                                                if 'thickness' in key.lower() or 'varying' in key.lower():
                                                    params_grp[key] = gt_grp[key][()]
                                                    # Copy attributes
                                                    for attr_name, attr_value in gt_grp[key].attrs.items():
                                                        params_grp[key].attrs[attr_name] = attr_value
                                    
                                    logging.info(f"    Saved reference and varying parameters only")
                                    continue
                        
                        # Full copy for first experiment or different ground truth
                        copy_hdf5_tree(output_file, exp_group, output_name)
                        
                    except Exception as e:
                        logging.warning(f"  Failed to copy {output_name}: {e}")
                
                # Save first experiment's output files for deduplication
                if i == 0:
                    first_exp_output_files = output_files.copy()
                
            except Exception as e:
                logging.error(f"  Failed to gather files from worktree for {exp_name}: {e}")
            
            finally:
                # Clean up the worktree now that we're done with this experiment
                if worktree_dir and worktree_dir.exists():
                    try:
                        logging.debug(f"  Cleaning up worktree {worktree_dir}")
                        cleanup_worktree(worktree_dir)
                    except Exception as e:
                        logging.debug(f"  Could not remove worktree: {e}")
    
    logging.info(f"Successfully created {output_path}")


def _copy_dict_to_hdf5(data: Dict, group: h5py.Group):
    """Recursively copy dictionary to HDF5 group.
    
    Args:
        data: Dictionary to copy
        group: HDF5 group to copy into
    """
    for key, value in data.items():
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            _copy_dict_to_hdf5(value, subgroup)
        elif isinstance(value, (list, tuple)):
            group.create_dataset(key, data=np.array(value))
        elif isinstance(value, np.ndarray):
            group.create_dataset(key, data=value)
        else:
            try:
                group.create_dataset(key, data=value)
            except (TypeError, ValueError):
                # Store as string if we can't store the type directly
                group.create_dataset(key, data=str(value))


def load_sweep_registry() -> Dict[str, Any]:
    """Load sweep registry from dvc_sweeps.yaml.
    
    Returns:
        Dictionary of sweep configurations
    """
    sweep_file = Path("dvc_sweeps.yaml")
    
    if not sweep_file.exists():
        return {}
    
    with open(sweep_file, 'r') as f:
        return yaml.load(f) or {}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Gather and pool DVC experiment results"
    )
    parser.add_argument(
        "sweep_id",
        nargs="?",
        help="Sweep ID to gather experiments for"
    )
    parser.add_argument(
        "--stage",
        help="Stage name to gather experiments for"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HDF5 file path (default: experiment_sweep_<sweep_id>.hdf5)"
    )
    parser.add_argument(
        "--log-dirs",
        nargs="+",
        type=Path,
        default=[Path("logs"), Path(Path.home() / "reconstructions")],
        help="Base directories to search for experiment logs"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Apply, commit, and push all experiments in the sweep to git/DVC remotes"
    )
    parser.add_argument(
        "--git-remote",
        default="origin",
        help="Git remote name for pushing (default: origin)"
    )
    parser.add_argument(
        "--dvc-remote",
        default=None,
        help="DVC remote name for pushing. If not specified, DVC will use its default remote."
    )
    parser.add_argument(
        "--include-stages",
        nargs="+",
        help="Only include these DVC stages (e.g., SI0009_reconstruct). If not specified, all stages are included."
    )
    parser.add_argument(
        "--exclude-stages",
        nargs="+",
        help="Exclude these DVC stages (e.g., SI0009_simulate to save space)"
    )
    parser.add_argument(
        "--split-by-stage",
        action="store_true",
        help="Create separate HDF5 files for each stage instead of a single pooled file"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear DVC cache after gathering (WARNING: this will delete cached experiment outputs, making re-gathering impossible)"
    )
    
    args = parser.parse_args()
    
    # Clean up any stale worktrees from previous interrupted runs
    cleanup_all_stale_worktrees()
    
    # Validate arguments
    if not args.sweep_id and not args.stage:
        parser.error("Either sweep_id or --stage must be provided")
    
    # Get experiments
    if args.sweep_id:
        experiments = get_experiments_by_sweep_id(args.sweep_id)
        sweep_id = args.sweep_id
    else:
        experiments = get_experiments_by_stage(args.stage)
        sweep_id = args.stage
    
    if not experiments:
        logging.error("No experiments found")
        sys.exit(1)
    
    # Set default output path
    if not args.output:
        args.output = Path(f"experiment_sweep_{sweep_id}.hdf5")
    
    # Write to /tmp first to avoid disk quota issues, then move to final location
    final_output = args.output
    temp_output = Path(f"/tmp/experiment_sweep_{sweep_id}.hdf5")
    
    logging.info(f"Final output will be: {final_output}")
    logging.info(f"Writing to temporary location: {temp_output}")
    
    # PASS 1: Gather metadata only (no worktrees - faster and minimal disk usage)
    logging.info("\n" + "="*60)
    logging.info("PASS 1: Gathering experiment metadata")
    logging.info("="*60 + "\n")
    
    experiments_metadata = []
    for exp in experiments:
        exp_meta = gather_experiment_metadata(exp, args.log_dirs, sweep_id)
        if exp_meta:
            experiments_metadata.append(exp_meta)
    
    if not experiments_metadata:
        logging.error("No experiment metadata could be gathered")
        sys.exit(1)
    
    logging.info(f"Gathered metadata for {len(experiments_metadata)} experiments")
    
    # Determine which stages to process
    if args.split_by_stage:
        # Collect all unique stages across experiments
        all_unique_stages = set()
        for exp_meta in experiments_metadata:
            all_unique_stages.update(exp_meta.get("all_stages", []))
        
        # Apply filters
        stages_to_process = filter_stages(
            sorted(all_unique_stages),
            include_stages=args.include_stages,
            exclude_stages=args.exclude_stages,
        )
        
        if not stages_to_process:
            logging.error("No stages to process after filtering")
            sys.exit(1)
        
        logging.info(f"Will create separate files for {len(stages_to_process)} stage(s): {', '.join(stages_to_process)}")
    else:
        stages_to_process = [None]  # Single file with all stages
        if args.include_stages or args.exclude_stages:
            logging.info("Stage filters will be applied to the single output file")
    
    # PASS 2: Stream worktree creation and file copying
    logging.info("\n" + "="*60)
    logging.info("PASS 2: Streaming file copy (one worktree at a time)")
    logging.info("="*60 + "\n")
    
    # Process each stage (or all stages together if not splitting)
    output_files_created = []
    
    for stage_filter in stages_to_process:
        if args.split_by_stage:
            # Create separate file for this stage
            stage_suffix = f"_{stage_filter}" if stage_filter else ""
            
            # Respect --output flag if provided, otherwise use default naming
            if args.output:
                # Use provided output name as base, add stage suffix
                base_name = args.output.stem
                current_temp_output = Path(f"/tmp/{base_name}{stage_suffix}.hdf5")
                current_final_output = Path(f"{base_name}{stage_suffix}.hdf5")
            else:
                # Default naming scheme
                current_temp_output = Path(f"/tmp/experiment_sweep_{sweep_id}{stage_suffix}.hdf5")
                current_final_output = Path(f"experiment_sweep_{sweep_id}{stage_suffix}.hdf5")
            
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing stage: {stage_filter}")
            logging.info(f"{'='*60}\n")
            
            # For split mode, only include this specific stage
            current_include = [stage_filter] if stage_filter else None
            current_exclude = None
        else:
            # Single file with all stages
            current_temp_output = temp_output
            current_final_output = final_output
            current_include = args.include_stages
            current_exclude = args.exclude_stages
        
        # Create pooled HDF5 file with streaming worktree creation (in /tmp)
        create_pooled_hdf5_streaming(
            experiments,
            experiments_metadata,
            current_temp_output,
            sweep_id,
            args.log_dirs,
            include_stages=current_include,
            exclude_stages=current_exclude,
        )
        
        output_files_created.append((current_temp_output, current_final_output))
    
    # PASS 3: Clear DVC cache and move files to final location
    logging.info("\n" + "="*60)
    logging.info("PASS 3: Clearing DVC cache and moving files to final location")
    logging.info("="*60 + "\n")
    
    # Check files were created successfully
    for temp_out, _ in output_files_created:
        if not temp_out.exists():
            logging.error(f"Temporary file {temp_out} was not created!")
            sys.exit(1)
        
        temp_size = temp_out.stat().st_size / (1024**3)
        logging.info(f"Created {temp_out.name}: {temp_size:.1f} GB")
    
    # Clear DVC cache to free up space (only if requested)
    if args.clear_cache:
        logging.info("Clearing DVC cache to free up disk space...")
        logging.warning("⚠ This will delete cached experiment outputs!")
        try:
            # Use gc with --workspace to keep only what's in current workspace
            # This will remove cached files for queued experiments
            result = subprocess.run(
                ["dvc", "gc", "--workspace", "--force"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logging.info("Successfully cleared DVC cache")
                # Show how much space was freed
                result = subprocess.run(
                    ["du", "-sh", str(Path.home() / "dvc_cache")],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    cache_size = result.stdout.strip().split()[0]
                    logging.info(f"DVC cache size after cleanup: {cache_size}")
            else:
                logging.warning(f"DVC gc failed: {result.stderr}")
        except Exception as e:
            logging.warning(f"Could not clear DVC cache: {e}")
    else:
        logging.info("Skipping DVC cache cleanup (outputs remain available for re-gathering)")
    
    # Move files from /tmp to final location
    for temp_out, final_out in output_files_created:
        logging.info(f"Moving {temp_out} to {final_out}...")
        try:
            # Use shutil.move for cross-filesystem move if needed
            shutil.move(str(temp_out), str(final_out))
            logging.info(f"Successfully moved file to {final_out}")
        except Exception as e:
            logging.error(f"Failed to move file: {e}")
            logging.error(f"Temporary file remains at: {temp_out}")
            sys.exit(1)
    
    # Add gathered files to DVC and save as experiment
    logging.info("\n" + "="*60)
    logging.info("Adding gathered files to DVC")
    logging.info("="*60 + "\n")
    
    for _, final_out in output_files_created:
        try:
            # Add the gathered HDF5 file to DVC tracking
            logging.info(f"Adding {final_out} to DVC tracking...")
            result = subprocess.run(
                ["dvc", "add", str(final_out)],
                capture_output=True,
                text=True,
                check=True,
            )
            logging.info(f"Successfully added {final_out} to DVC")
            
            # Stage the .dvc file for git
            dvc_file = Path(f"{final_out}.dvc")
            if dvc_file.exists():
                logging.info(f"Staging {dvc_file} for git...")
                subprocess.run(
                    ["git", "add", str(dvc_file), ".gitignore"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                
        except subprocess.CalledProcessError as e:
            logging.error(f"Error adding {final_out} to DVC: {e.stderr}")
        except Exception as e:
            logging.error(f"Error: {e}")
    
    # Save the workspace state as a DVC experiment with the sweep name (once for all files)
    try:
        logging.info(f"Saving workspace as DVC experiment '{sweep_id}'...")
        result = subprocess.run(
            ["dvc", "exp", "save", "-n", sweep_id, "-f"],
            capture_output=True,
            text=True,
        )
        
        if result.returncode == 0:
            logging.info(f"Successfully saved workspace as experiment '{sweep_id}'")
            logging.info("You can later recover this sweep with: dvc exp apply <sweep_id>")
        else:
            logging.warning(f"Could not save as experiment: {result.stderr}")
            
    except subprocess.CalledProcessError as e:
        logging.error(f"Error saving experiment: {e.stderr}")
    except Exception as e:
        logging.error(f"Error: {e}")
    
    # Optionally apply, commit, and push experiments to remote
    if args.push:
        logging.info("\n" + "="*60)
        logging.info("Applying, committing, and pushing experiments to remote")
        logging.info("="*60 + "\n")
        apply_commit_push_experiments(experiments, sweep_id, args.git_remote, args.dvc_remote)
    
    logging.info("Done!")


if __name__ == "__main__":
    main()
