"""Policies: abstract base class and concrete implementations."""

import collections
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
import torch as th
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.distributions import (
    BernoulliDistribution, CategoricalDistribution, DiagGaussianDistribution,
    Distribution, MultiCategoricalDistribution,
    StateDependentNoiseDistribution, make_proba_distribution)
from stable_baselines3.common.preprocessing import (get_action_dim,
                                                    is_image_space,
                                                    preprocess_obs)
from stable_baselines3.common.torch_layers import (BaseFeaturesExtractor,
                                                   FlattenExtractor,
                                                   MlpExtractor, NatureCNN,
                                                   create_mlp)
from stable_baselines3.common.utils import (get_device,
                                            is_vectorized_observation)
from stable_baselines3.common.vec_env import VecTransposeImage


class BaseModel(nn.Module, ABC):
    """
    The base model object: makes predictions in response to observations.

    In the case of policies, the prediction is an action. In the case of critics, it is the
    estimated value of the observation.

    :param observation_space: The observation space of the environment
    :param action_space: The action space of the environment
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the feature extractor.
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        features_extractor: Optional[nn.Module] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(BaseModel, self).__init__()

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.normalize_images = normalize_images

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = None  # type: Optional[th.optim.Optimizer]
        self.optimizer_QN = None  # type: Optional[th.optim.Optimizer]

        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs

    @abstractmethod
    def forward(self, *args, **kwargs):
        del args, kwargs

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No feature extractor was set"
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.features_extractor(preprocessed_obs)

    def _get_data(self) -> Dict[str, Any]:
        """
        Get data that need to be saved in order to re-create the model.
        This corresponds to the arguments of the constructor.

        :return:
        """
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            # Passed to the constructor by child class
            # squash_output=self.squash_output,
            # features_extractor=self.features_extractor
            normalize_images=self.normalize_images,
        )

    @property
    def device(self) -> th.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'auto' device is used as a fallback.

        :return:"""
        for param in self.parameters():
            return param.device
        return get_device("auto")

    def save(self, path: str) -> None:
        """
        Save model to a given location.

        :param path:
        """
        th.save({"state_dict": self.state_dict(), "data": self._get_data()}, path)

    @classmethod
    def load(cls, path: str, device: Union[th.device, str] = "auto") -> "BaseModel":
        """
        Load model from path.

        :param path:
        :param device: Device on which the policy should be loaded.
        :return:
        """
        device = get_device(device)
        saved_variables = th.load(path, map_location=device)
        # Create policy object
        model = cls(**saved_variables["data"])  # pytype: disable=not-instantiable
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model

    def load_from_vector(self, vector: np.ndarray):
        """
        Load parameters from a 1D vector.

        :param vector:
        """
        th.nn.utils.vector_to_parameters(th.FloatTensor(vector).to(self.device), self.parameters())

    def parameters_to_vector(self) -> np.ndarray:
        """
        Convert the parameters to a 1D vector.

        :return:
        """
        return th.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()


class BasePolicy(BaseModel):
    """The base policy object.

    Parameters are mostly the same as `BaseModel`; additions are documented below.

    :param args: positional arguments passed through to `BaseModel`.
    :param kwargs: keyword arguments passed through to `BaseModel`.
    :param squash_output: For continuous actions, whether the output is squashed
        or not using a ``tanh()`` function.
    """

    def __init__(self, *args, squash_output: bool = False, **kwargs):
        super(BasePolicy, self).__init__(*args, **kwargs)
        self._squash_output = squash_output

    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        """ (float) Useful for pickling policy."""
        del progress_remaining
        return 0.0

    @property
    def squash_output(self) -> bool:
        """(bool) Getter for squash_output."""
        return self._squash_output

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    @abstractmethod
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the policy action and state from an observation (and optional state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if mask is None:
        #     mask = [False for _ in range(self.n_envs)]
        observation = np.array(observation)

        # Handle the different cases for images
        # as PyTorch use channel first format
        if is_image_space(self.observation_space):
            if not (
                observation.shape == self.observation_space.shape or observation.shape[1:] == self.observation_space.shape
            ):
                # Try to re-order the channels
                transpose_obs = VecTransposeImage.transpose_image(observation)
                if (
                    transpose_obs.shape == self.observation_space.shape
                    or transpose_obs.shape[1:] == self.observation_space.shape
                ):
                    observation = transpose_obs

        vectorized_env = is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)

        observation = th.as_tensor(observation).to(self.device)
        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions = actions[0]

        return actions, state

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def idx2vector(self, indices, height, width):
        vector_all = []
        if isinstance(indices, torch.Tensor):
            for idx in indices:
                map = np.zeros(shape=[height, width])
                x, y = int(torch.round(idx[0])), int(torch.round(idx[1]))
                # if x - idx[0] != 0:
                #     print('debug')
                map[x, y] = 1  # + idx[0] - x + idx[1] - y
                vector_all.append(map.flatten())
            return torch.Tensor(np.array(vector_all)).to(self.device)
        else:
            for idx in indices:
                map = np.zeros(shape=[height, width])
                x, y = int(round(idx[0], 0)), int(round(idx[1], 0))
                # if x - idx[0] != 0:
                #     print('debug')
                map[x, y] = 1  # + idx[0] - x + idx[1] - y
                vector_all.append(map.flatten())
            return np.asarray(vector_all)


class ActorCriticPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the feature extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(ActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [dict(pi=[64, 64], vf=[64, 64])]
            else:
                net_arch = []

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": sde_net_arch is not None,
            }

        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                sde_net_arch=default_none_kwargs["sde_net_arch"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn)

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Separate feature extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                self.features_dim, self.sde_net_arch, self.activation_fn
            )

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # feature_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        return latent_pi, latent_vf, latent_sde

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        latent_pi, _, latent_sde = self._get_latent(observation)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        return distribution.get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()


class ActorTwoCriticsPolicy(ActorCriticPolicy):
    """Implements the actor critic algorithm but with two value networks (for reward
    and cost)"""
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        recon_obs: bool = False,
        env_configs: dict = None,
        tau_update: float = 0.01,
    ):
        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [dict(pi=[64, 64], vf=[64, 64], cvf=[64,64])]
            else:
                net_arch = []

        super(ActorTwoCriticsPolicy, self).__init__(
              observation_space,
              action_space,
              lr_schedule,
              net_arch,
              activation_fn,
              ortho_init,
              use_sde,
              log_std_init,
              full_std,
              sde_net_arch,
              use_expln,
              squash_output,
              features_extractor_class,
              features_extractor_kwargs,
              normalize_images,
              optimizer_class,
              optimizer_kwargs
        )
        self.recon_obs = recon_obs
        self.tau_update = tau_update
        self.env_configs = env_configs
        self.recon_build(lr_schedule, recon_obs)

    def recon_build_mlp_extractor(self, recon_obs: bool = False) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        if recon_obs == True:
            self.features_dim = self.env_configs['map_height'] * self.env_configs['map_width']
        self.mlp_extractor = MlpExtractor(self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn,
                                          create_cvf=True)

    def recon_build(self, lr_schedule: Callable[[float], float], recon_obs: bool = False) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.recon_build_mlp_extractor(recon_obs)

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Separate feature extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                self.features_dim, self.sde_net_arch, self.activation_fn
            )

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.cost_value_net = nn.Linear(self.mlp_extractor.latent_dim_cvf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # feature_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
                self.cost_value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))
        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf, latent_cvf, latent_sde = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        cost_values = self.cost_value_net(latent_cvf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, cost_values, log_prob

    def get_cost_value(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        with th.no_grad():
            _, _, latent_cvf,_ = self._get_latent(obs)
            cost_values = self.cost_value_net(latent_cvf)
        return cost_values

    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: Observation
        :return: Latent codes
            for the actor, the value function, the cost value function and for gSDE function
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.recon_obs:
            features = self.idx2vector(features, height=self.env_configs['map_height'], width=self.env_configs['map_width'])
        latent_pi, latent_vf, latent_cvf = self.mlp_extractor(features)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        return latent_pi, latent_vf, latent_cvf, latent_sde

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, cost value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_vf, latent_cvf, latent_sde = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        cost_values = self.cost_value_net(latent_cvf)
        return values, cost_values, log_prob, distribution.entropy()

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        latent_pi, _, _, latent_sde = self._get_latent(observation)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        return distribution.get_actions(deterministic=deterministic)



class DistributionalActorTwoCriticsPolicy(ActorCriticPolicy):
    """Implements the actor critic algorithm but with two value networks (for reward
    and cost)"""
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        recon_obs: bool = False,
        env_configs: dict = None,
        N: int = 64,
        cost_quantile:int = 48,
        tau_update: float = 0.01,
        LR_QN: float = 0.001,
        qnet_layers: Optional[List[int]] = [256, 256],
        type: str = 'VaR',
        prob_yita = 0.01,
        method: str = None,
        device: Union[th.device, str] = "cuda",
        input_action: bool = True
    ):
        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [dict(pi=[64, 64], vf=[64, 64], cvf=[256,256])]
            else:
                net_arch = []

        super(DistributionalActorTwoCriticsPolicy, self).__init__(
              observation_space,
              action_space,
              lr_schedule,
              net_arch,
              activation_fn,
              ortho_init,
              use_sde,
              log_std_init,
              full_std,
              sde_net_arch,
              use_expln,
              squash_output,
              features_extractor_class,
              features_extractor_kwargs,
              normalize_images,
              optimizer_class,
              optimizer_kwargs,
        )
        self.recon_obs = recon_obs
        self.env_configs = env_configs
        self.cost_quantile = cost_quantile
        self.type = type
        self.prob_yita = prob_yita
        self.method = method
        self.input_action = input_action
        if method == 'QRDQN':
            self.dis_build_QRDQN(lr_schedule, N, tau_update, LR_QN, qnet_layers, recon_obs, input_action)
        elif method == 'IQN':
            self.dis_build_IQN(device, lr_schedule, N, tau_update, LR_QN, qnet_layers, recon_obs, input_action)
        elif method == 'SplineDQN':
            self.dis_build_SplineDQN(device, lr_schedule, N, tau_update, LR_QN, qnet_layers, recon_obs, input_action)
        elif method == 'NCQR':
            self.dis_build_NCQR(device, lr_schedule, N, tau_update, LR_QN, qnet_layers, recon_obs, input_action)



    def recon_build_mlp_extractor(self, recon_obs: bool = False) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        '''
        MlpExtractor(
            (shared_net): Sequential()
        (policy_net): Sequential(
            (0): Linear(in_features=18, out_features=64, bias=True)
        (1): Tanh()
        (2): Linear(in_features=64, out_features=64, bias=True)
        (3): Tanh()
        )
        (value_net): Sequential(
            (0): Linear(in_features=18, out_features=64, bias=True)
        (1): Tanh()
        (2): Linear(in_features=64, out_features=64, bias=True)
        (3): Tanh()
        )
        (cost_value_net): Sequential(
            (0): Linear(in_features=18, out_features=256, bias=True)
        (1): Tanh()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): Tanh()
        )
        )
        '''
        #  mlpextractor---------------------------------------------------------------------------------------------------------
        if recon_obs == True:
            self.features_dim = self.env_configs['map_height'] * self.env_configs['map_width']
        self.mlp_extractor = MlpExtractor(self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn,
                                          create_cvf=False)


    def dis_build_QRDQN(self, lr_schedule: Callable[[float], float], N, tau_update, LR_QN, qnet_layers,
                        recon_obs: bool = False, input_action:bool = True) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.recon_build_mlp_extractor(recon_obs)
        self.N = N
        # self.quantile_tau = th.FloatTensor([i / self.N for i in range(1, self.N + 1)])
        self.tau_update = tau_update
        self.LR_QN = LR_QN
        self.qnet_layers = qnet_layers
        self.input_action = input_action

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Separate feature extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                self.features_dim, self.sde_net_arch, self.activation_fn
            )
        #  action net--------------------------------------------------------------------------------------------------------------
        #  Linear(in_features=64, out_features=6, bias=True)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")
        # value net--------------------------------------------------------------------------------------------------------------
        # Linear(in_features=64, out_features=1, bias=True)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # cost net--------------------------------------------------------------------------------------------------------------
        # Linear(in_features=64, out_features=1, bias=True)

        action_dim = get_action_dim(self.action_space)

        # q_net_target = create_mlp(self.features_dim + action_dim, self.N, self.qnet_layers)
        # self.cost_value_net_target = nn.Sequential(*q_net_target)
        #
        # q_net_local = create_mlp(self.features_dim + action_dim, self.N, self.qnet_layers)
        # self.cost_value_net_local = nn.Sequential(*q_net_local)

        # if self.input_action:
        #     self.cost_value_net_local = QRDQN(self.features_dim + action_dim, self.qnet_layers, self.N)
        #     self.cost_value_net_target = QRDQN(self.features_dim + action_dim, self.qnet_layers, self.N)
        # else:
        #     self.cost_value_net_local = QRDQN(self.features_dim, self.qnet_layers, self.N)
        #     self.cost_value_net_target = QRDQN(self.features_dim, self.qnet_layers, self.N)

        self.cost_value_net_local = QRDQN(self.qnet_layers, self.features_dim, self.N)
        self.cost_value_net_target = QRDQN(self.qnet_layers, self.features_dim, self.N)

        # self.cost_value_net_target = nn.Linear(self.mlp_extractor.latent_dim_cvf, self.N)
        # self.cost_value_net_local = nn.Linear(self.mlp_extractor.latent_dim_cvf, self.N)

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # feature_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
                # self.cost_value_net_target: np.sqrt(2),
                # self.cost_value_net_local: np.sqrt(2),
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        # self.optimizer_QN = th.optim.Adam(self.cost_value_net_local.parameters(), lr=self.LR_QN)
        # self.optimizer_QN = self.optimizer_class(self.cost_value_net_local.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def dis_build_IQN(self, device, lr_schedule: Callable[[float], float], N, tau_update, LR_QN,
                      qnet_layers, recon_obs: bool = False, input_action:bool = True) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.recon_build_mlp_extractor(recon_obs)
        self.N = N
        self.n_cos = 64
        self.tau_update = tau_update
        self.LR_QN = LR_QN
        self.qnet_layers = qnet_layers
        self.input_action = input_action

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Separate feature extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                self.features_dim, self.sde_net_arch, self.activation_fn
            )
        #  action net--------------------------------------------------------------------------------------------------------------
        #  Linear(in_features=64, out_features=6, bias=True)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")
        # value net--------------------------------------------------------------------------------------------------------------
        # Linear(in_features=64, out_features=1, bias=True)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # cost net--------------------------------------------------------------------------------------------------------------
        # Linear(in_features=64, out_features=1, bias=True)

        action_dim = get_action_dim(self.action_space)

        # self.head = th.nn.Linear(self.features_dim + action_dim, self.qnet_layers[0])
        # self.cos_embedding = th.nn.Linear(self.n_cos, self.qnet_layers[0])
        # self.ff_1 = nn.Linear(self.qnet_layers[0], self.qnet_layers[1])
        # self.ff_2 = nn.Linear(self.qnet_layers[0], 1)
        #def __init__(self, state_size, layer_size, n_cos: int = 64):

        # if self.input_action:
        #     self.cost_value_net_local = IQN(self.features_dim + action_dim, self.qnet_layers, self.n_cos, self.N, device)
        #     self.cost_value_net_target = IQN(self.features_dim + action_dim, self.qnet_layers, self.n_cos, self.N, device)
        # else:
        #     self.cost_value_net_local = IQN(self.features_dim, self.qnet_layers, self.n_cos, self.N, device)
        #     self.cost_value_net_target = IQN(self.features_dim, self.qnet_layers, self.n_cos, self.N, device)
        self.cost_value_net_local = IQN(self.qnet_layers, self.features_dim, self.N, device)
        self.cost_value_net_target = IQN(self.qnet_layers, self.features_dim, self.N, device)

        # self.cost_value_net_target = nn.Linear(self.mlp_extractor.latent_dim_cvf, self.N)
        # self.cost_value_net_local = nn.Linear(self.mlp_extractor.latent_dim_cvf, self.N)

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # feature_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
                # self.cost_value_net_target: np.sqrt(2),
                # self.cost_value_net_local: np.sqrt(2),
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        # self.optimizer_QN = th.optim.Adam(self.cost_value_net_local.parameters(), lr=self.LR_QN)
        # self.optimizer_QN = self.optimizer_class(self.cost_value_net_local.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def dis_build_SplineDQN(self, device, lr_schedule: Callable[[float], float], N, tau_update, LR_QN,
                      qnet_layers, recon_obs: bool = False, input_action:bool = True) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.recon_build_mlp_extractor(recon_obs)
        self.N = N
        self.n_cos = 64
        self.tau_update = tau_update
        self.LR_QN = LR_QN
        self.qnet_layers = qnet_layers
        self.input_action = input_action

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Separate feature extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                self.features_dim, self.sde_net_arch, self.activation_fn
            )
        #  action net--------------------------------------------------------------------------------------------------------------
        #  Linear(in_features=64, out_features=6, bias=True)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")
        # value net--------------------------------------------------------------------------------------------------------------
        # Linear(in_features=64, out_features=1, bias=True)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # cost net--------------------------------------------------------------------------------------------------------------
        # Linear(in_features=64, out_features=1, bias=True)

        action_dim = get_action_dim(self.action_space)

        # self.head = th.nn.Linear(self.features_dim + action_dim, self.qnet_layers[0])
        # self.cos_embedding = th.nn.Linear(self.n_cos, self.qnet_layers[0])
        # self.ff_1 = nn.Linear(self.qnet_layers[0], self.qnet_layers[1])
        # self.ff_2 = nn.Linear(self.qnet_layers[0], 1)
        #def __init__(self, state_size, layer_size, n_cos: int = 64):

        # if self.input_action:
        #     self.cost_value_net_local = IQN(self.features_dim + action_dim, self.qnet_layers, self.n_cos, self.N, device)
        #     self.cost_value_net_target = IQN(self.features_dim + action_dim, self.qnet_layers, self.n_cos, self.N, device)
        # else:
        #     self.cost_value_net_local = IQN(self.features_dim, self.qnet_layers, self.n_cos, self.N, device)
        #     self.cost_value_net_target = IQN(self.features_dim, self.qnet_layers, self.n_cos, self.N, device)
        self.cost_value_net_local = SplineDQN(self.qnet_layers, self.features_dim, self.N, device)
        #self.cost_value_net_target = SplineDQN(self.qnet_layers, self.features_dim, self.N, device)

        # self.cost_value_net_target = nn.Linear(self.mlp_extractor.latent_dim_cvf, self.N)
        # self.cost_value_net_local = nn.Linear(self.mlp_extractor.latent_dim_cvf, self.N)

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # feature_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
                # self.cost_value_net_target: np.sqrt(2),
                # self.cost_value_net_local: np.sqrt(2),
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        # self.optimizer_QN = th.optim.Adam(self.cost_value_net_local.parameters(), lr=self.LR_QN)
        # self.optimizer_QN = self.optimizer_class(self.cost_value_net_local.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def dis_build_NCQR(self, device, lr_schedule: Callable[[float], float], N, tau_update, LR_QN,
                      qnet_layers, recon_obs: bool = False, input_action:bool = True) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.recon_build_mlp_extractor(recon_obs)
        self.N = N
        self.n_cos = 64
        self.tau_update = tau_update
        self.LR_QN = LR_QN
        self.qnet_layers = qnet_layers
        self.input_action = input_action

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Separate feature extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                self.features_dim, self.sde_net_arch, self.activation_fn
            )
        #  action net--------------------------------------------------------------------------------------------------------------
        #  Linear(in_features=64, out_features=6, bias=True)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")
        # value net--------------------------------------------------------------------------------------------------------------
        # Linear(in_features=64, out_features=1, bias=True)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # cost net--------------------------------------------------------------------------------------------------------------
        # Linear(in_features=64, out_features=1, bias=True)

        action_dim = get_action_dim(self.action_space)

        # self.head = th.nn.Linear(self.features_dim + action_dim, self.qnet_layers[0])
        # self.cos_embedding = th.nn.Linear(self.n_cos, self.qnet_layers[0])
        # self.ff_1 = nn.Linear(self.qnet_layers[0], self.qnet_layers[1])
        # self.ff_2 = nn.Linear(self.qnet_layers[0], 1)
        #def __init__(self, state_size, layer_size, n_cos: int = 64):

        # if self.input_action:
        #     self.cost_value_net_local = IQN(self.features_dim + action_dim, self.qnet_layers, self.n_cos, self.N, device)
        #     self.cost_value_net_target = IQN(self.features_dim + action_dim, self.qnet_layers, self.n_cos, self.N, device)
        # else:
        #     self.cost_value_net_local = IQN(self.features_dim, self.qnet_layers, self.n_cos, self.N, device)
        #     self.cost_value_net_target = IQN(self.features_dim, self.qnet_layers, self.n_cos, self.N, device)
        self.cost_value_net_local = NCQR(self.qnet_layers, self.features_dim, self.N, device)
        self.cost_value_net_target = NCQR(self.qnet_layers, self.features_dim, self.N, device)

        # self.cost_value_net_target = nn.Linear(self.mlp_extractor.latent_dim_cvf, self.N)
        # self.cost_value_net_local = nn.Linear(self.mlp_extractor.latent_dim_cvf, self.N)

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # feature_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
                # self.cost_value_net_target: np.sqrt(2),
                # self.cost_value_net_local: np.sqrt(2),
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        # self.optimizer_QN = th.optim.Adam(self.cost_value_net_local.parameters(), lr=self.LR_QN)
        # self.optimizer_QN = self.optimizer_class(self.cost_value_net_local.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]: #important-----------------
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        # get actor distribution from action net
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        # sample
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        # cost_values = self.cost_value_net(feature+action)
        features = self.extract_features(obs)
        if self.recon_obs:
            features = self.idx2vector(features, height=self.env_configs['map_height'], width=self.env_configs['map_width'])


        if self.method == 'QRDQN' or self.method=='SplineDQN' or self.method=='NCQR':
            distributional_cost_values = self.cost_value_net_local(features)
            # if self.input_action:
            #     distributional_cost_values = self.cost_value_net_local(th.cat([features, actions], dim=1))
            # else:
            #     distributional_cost_values = self.cost_value_net_local(features)

        elif self.method == 'IQN':
            # if self.input_action:
            #     distributional_cost_values, _ = self.cost_value_net_local(th.cat([features, actions], dim=1))
            # else:
            #     distributional_cost_values, _ = self.cost_value_net_local(features)
            distributional_cost_values, _ = self.cost_value_net_local(features)

        if self.type == 'VaR':
            # Caculate the cost values using VaR method
            cost_values = distributional_cost_values[:,self.cost_quantile-1].view(distributional_cost_values.shape[0], 1)
        elif self.type == 'CVaR':
            # Caculate the cost values using CVaR method
            # VaR = distributional_cost_values[:, self.cost_quantile - 1].view(distributional_cost_values.shape[0], 1)
            # cost_values = torch.zeros(distributional_cost_values.shape[0], 1).to(self.device)
            # for i in range(0, distributional_cost_values.shape[0]):
            #     selected_cost_values = (th.where(distributional_cost_values[i] < VaR[i], 0, distributional_cost_values[i]))
            #     #selected_cost_values = selected_cost_values.view(-1, distributional_cost_values.shape[1])
            #     non_zero_selected_cost_values = selected_cost_values[selected_cost_values.nonzero(as_tuple=True)]
            #     avg_cost_values = non_zero_selected_cost_values.mean()
            #     cost_values[i] = avg_cost_values

            # VaR = distributional_cost_values[:, self.cost_quantile - 1].view(distributional_cost_values.shape[0], 1)
            # sum = (torch.zeros(distributional_cost_values.shape[0], 1)).to(self.device)
            # num = (torch.zeros(distributional_cost_values.shape[0], 1)).to(self.device)
            # cost_values = (torch.zeros(distributional_cost_values.shape[0], 1)).to(self.device)
            #
            # for i in range (0, distributional_cost_values.shape[0]):
            #     for quant in range(0, self.N):
            #         quant_value = distributional_cost_values[i, quant]
            #         if quant_value >= VaR[i]:
            #             sum[i] = sum[i] + quant_value
            #             num[i] = num[i] + 1
            #     cost_values[i] = sum[i] / num[i]

            VaR = distributional_cost_values[:, self.cost_quantile - 1].view(distributional_cost_values.shape[0], 1)
            alpha = self.cost_quantile / self.N
            exp = th.mean(abs(distributional_cost_values - VaR), dim=1).view(distributional_cost_values.shape[0], 1)
            cost_values = VaR + exp / (1.0-alpha)
        elif self.type == 'Prob':

            num = torch.zeros(distributional_cost_values.shape[0], 1)
            cost_values = torch.zeros(distributional_cost_values.shape[0], 1)

            for i in range(0, distributional_cost_values.shape[0]):
                for quant in range(0, self.N):
                    quant_value = distributional_cost_values[i, quant]
                    if quant_value >= self.prob_yita:
                        num[i] = num[i] + 1
                cost_values[i] = num[i] *1.0 / self.N
        elif self.type == 'Expectation':
            cost_values = torch.mean(distributional_cost_values, dim=1).view(distributional_cost_values.shape[0], 1)

        return actions, values, cost_values, log_prob

    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: Observation
        :return: Latent codes
            for the actor, the value function, the cost value function and for gSDE function
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.recon_obs:
            features = self.idx2vector(features, height=self.env_configs['map_height'], width=self.env_configs['map_width'])
        latent_pi, latent_vf= self.mlp_extractor(features)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        return latent_pi, latent_vf, latent_sde

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, cost value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        # cost_values = self.cost_value_net(feature+action)

        features = self.extract_features(obs)
        if self.recon_obs:
            features = self.idx2vector(features, height=self.env_configs['map_height'], width=self.env_configs['map_width'])


        if self.method == 'QRDQN' or self.method=='SplineDQN' or self.method=='NCQR':
            # if self.input_action:
            #     distributional_cost_values = self.cost_value_net_local(th.cat([features, actions], dim=1))
            # else:
            #     distributional_cost_values = self.cost_value_net_local(features)
            distributional_cost_values = self.cost_value_net_local(features)
        elif self.method == 'IQN':
            # if self.input_action:
            #     distributional_cost_values, _ = self.cost_value_net_local(th.cat([features, actions], dim=1))
            # else:
            #     distributional_cost_values, _ = self.cost_value_net_local(features)
            distributional_cost_values, _ = self.cost_value_net_local(features)

        # cost_values = distributional_cost_values[:,self.cost_quantile-1]
        # cost_values = cost_values.view(distributional_cost_values.shape[0], 1)
        if self.type == 'VaR':
            # Caculate the cost values using VaR method
            cost_values = distributional_cost_values[:,self.cost_quantile-1].view(distributional_cost_values.shape[0], 1)
        elif self.type == 'CVaR':
            # Caculate the cost values using CVaR method
            VaR = distributional_cost_values[:, self.cost_quantile - 1].view(distributional_cost_values.shape[0], 1)
            alpha = self.cost_quantile / self.N
            exp = th.mean(abs(distributional_cost_values - VaR), dim=1).view(distributional_cost_values.shape[0], 1)
            cost_values = VaR + exp / (1.0-alpha)
            # VaR = distributional_cost_values[:, self.cost_quantile - 1].view(distributional_cost_values.shape[0], 1)
            # cost_values = torch.zeros(distributional_cost_values.shape[0], 1).to(self.device)
            # for i in range(0, distributional_cost_values.shape[0]):
            #     selected_cost_values = (th.where(distributional_cost_values[i] < VaR[i], 0, distributional_cost_values[i]))
            #     #selected_cost_values = selected_cost_values.view(-1, distributional_cost_values.shape[1])
            #     non_zero_selected_cost_values = selected_cost_values[selected_cost_values.nonzero(as_tuple=True)]
            #     avg_cost_values = non_zero_selected_cost_values.mean()
            #     cost_values[i] = avg_cost_values

            # VaR = distributional_cost_values[:, self.cost_quantile - 1].view(distributional_cost_values.shape[0], 1)
            # sum = (torch.zeros(distributional_cost_values.shape[0], 1)).to(self.device)
            # num = (torch.zeros(distributional_cost_values.shape[0], 1)).to(self.device)
            # cost_values = (torch.zeros(distributional_cost_values.shape[0], 1)).to(self.device)
            #
            # for i in range (0, distributional_cost_values.shape[0]):
            #     for quant in range(0, self.N):
            #         quant_value = distributional_cost_values[i, quant]
            #         if quant_value >= VaR[i]:
            #             sum[i] = sum[i] + quant_value
            #             num[i] = num[i] + 1
            #     cost_values[i] = sum[i] / num[i]

            # cost_values = distributional_cost_values[:,self.cost_quantile-1: self.N]
            # cost_values = th.mean(cost_values, dim=1)
            # cost_values = cost_values.view(distributional_cost_values.shape[0], 1)
        elif self.type == 'Prob':

            num = torch.zeros(distributional_cost_values.shape[0], 1)
            cost_values = torch.zeros(distributional_cost_values.shape[0], 1)

            for i in range(0, distributional_cost_values.shape[0]):
                for quant in range(0, self.N):
                    quant_value = distributional_cost_values[i, quant]
                    if quant_value >= self.prob_yita:
                        num[i] = num[i] + 1
                cost_values[i] = num[i] *1.0 / self.N
        elif self.type == 'Expectation':
            cost_values = torch.mean(distributional_cost_values, dim=1).view(distributional_cost_values.shape[0], 1)

        return values, cost_values, log_prob, distribution.entropy()

    def get_cost_distribution(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]: #important-----------------
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        with th.no_grad():
            latent_pi, _, latent_sde = self._get_latent(obs)
            # get actor distribution from action net
            distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
            # sample
            actions = distribution.get_actions(deterministic=deterministic)

            # cost_values = self.cost_value_net(feature+action)
            features = self.extract_features(obs)
            if self.recon_obs:
                features = self.idx2vector(features, height=self.env_configs['map_height'], width=self.env_configs['map_width'])


            if self.method == 'QRDQN' or self.method=='SplineDQN' or self.method=='NCQR':
                # if self.input_action:
                #     distributional_cost_values = self.cost_value_net_local(th.cat([features, actions], dim=1))
                # else:
                #     distributional_cost_values = self.cost_value_net_local(features)
                distributional_cost_values = self.cost_value_net_local(features)
            elif self.method == 'IQN':
                # if self.input_action:
                #     distributional_cost_values, _ = self.cost_value_net_local(th.cat([features, actions], dim=1))
                # else:
                #     distributional_cost_values, _ = self.cost_value_net_local(features)
                distributional_cost_values, _ = self.cost_value_net_local(features)

            # Caculate the cost values using VaR method
            cost_values_var = distributional_cost_values[:, self.cost_quantile - 1].view(distributional_cost_values.shape[0], 1)

            # Caculate the cost values using CVaR method
            VaR = cost_values_var
            alpha = self.cost_quantile / self.N
            exp = th.mean(abs(distributional_cost_values - VaR), dim=1).view(distributional_cost_values.shape[0], 1)
            cost_values_cvar = VaR + exp / (1.0-alpha)

            # Caculate the cost values using Expectation method
            cost_values_exp = torch.mean(distributional_cost_values, dim=1).view(distributional_cost_values.shape[0], 1)

        return distributional_cost_values, cost_values_var, cost_values_cvar, cost_values_exp

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        latent_pi, _, latent_sde = self._get_latent(observation)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        return distribution.get_actions(deterministic=deterministic)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class ActorCriticCnnPolicy(ActorCriticPolicy):
    """
    CNN policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the feature extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        super(ActorCriticCnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

class ActorTwoCriticsCnnPolicy(ActorTwoCriticsPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        super(ActorTwoCriticsCnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


class ContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        with th.no_grad():
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs)
        return self.q_networks[0](th.cat([features, actions], dim=1))


def create_sde_features_extractor(
    features_dim: int, sde_net_arch: List[int], activation_fn: Type[nn.Module]
) -> Tuple[nn.Sequential, int]:
    """
    Create the neural network that will be used to extract features
    for the gSDE exploration function.

    :param features_dim:
    :param sde_net_arch:
    :param activation_fn:
    :return:
    """
    # Special case: when using states as features (i.e. sde_net_arch is an empty list)
    # don't use any activation function
    sde_activation = activation_fn if len(sde_net_arch) > 0 else None
    latent_sde_net = create_mlp(features_dim, -1, sde_net_arch, activation_fn=sde_activation, squash_output=False)
    latent_sde_dim = sde_net_arch[-1] if len(sde_net_arch) > 0 else features_dim
    sde_features_extractor = nn.Sequential(*latent_sde_net)
    return sde_features_extractor, latent_sde_dim


_policy_registry = dict()  # type: Dict[Type[BasePolicy], Dict[str, Type[BasePolicy]]]


def get_policy_from_name(base_policy_type: Type[BasePolicy], name: str) -> Type[BasePolicy]:
    """
    Returns the registered policy from the base type and name.
    See `register_policy` for registering policies and explanation.

    :param base_policy_type: the base policy class
    :param name: the policy name
    :return: the policy
    """
    if base_policy_type not in _policy_registry:
        raise KeyError(f"Error: the policy type {base_policy_type} is not registered!")
    if name not in _policy_registry[base_policy_type]:
        raise KeyError(
            f"Error: unknown policy type {name},"
            f"the only registed policy type are: {list(_policy_registry[base_policy_type].keys())}!"
        )
    return _policy_registry[base_policy_type][name]


def register_policy(name: str, policy: Type[BasePolicy]) -> None:
    """
    Register a policy, so it can be called using its name.
    e.g. SAC('MlpPolicy', ...) instead of SAC(MlpPolicy, ...).

    The goal here is to standardize policy naming, e.g.
    all algorithms can call upon "MlpPolicy" or "CnnPolicy",
    and they receive respective policies that work for them.
    Consider following:

    OnlinePolicy
    -- OnlineMlpPolicy ("MlpPolicy")
    -- OnlineCnnPolicy ("CnnPolicy")
    OfflinePolicy
    -- OfflineMlpPolicy ("MlpPolicy")
    -- OfflineCnnPolicy ("CnnPolicy")

    Two policies have name "MlpPolicy" and two have "CnnPolicy".
    In `get_policy_from_name`, the parent class (e.g. OnlinePolicy)
    is given and used to select and return the correct policy.

    :param name: the policy name
    :param policy: the policy class
    """
    sub_class = None
    for cls in BasePolicy.__subclasses__():
        if issubclass(policy, cls):
            sub_class = cls
            break
    if sub_class is None:
        raise ValueError(f"Error: the policy {policy} is not of any known subclasses of BasePolicy!")

    if sub_class not in _policy_registry:
        _policy_registry[sub_class] = {}
    if name in _policy_registry[sub_class]:
        # Check if the registered policy is same
        # we try to register. If not so,
        # do not override and complain.
        if _policy_registry[sub_class][name] != policy:
            raise ValueError(f"Error: the name {name} is already registered for a different policy, will not override.")
    _policy_registry[sub_class][name] = policy

def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)

WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4

# class QRDQN(nn.Module):
#     def __init__(self, state_size, layer_size, N, seed: int=0):
#         super(QRDQN, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.input_shape = state_size
#         self.N = N
#
#         self.head_1 = nn.Linear(self.input_shape, layer_size[0])
#         self.ff_1 = nn.Linear(layer_size[0], layer_size[1])
#         self.ff_2 = nn.Linear(layer_size[1], self.N)
#         weight_init([self.head_1, self.ff_1, self.ff_2])
#
#     def forward(self, input):
#
#         x = torch.relu(self.head_1(input))
#         x = torch.relu(self.ff_1(x))
#         out = self.ff_2(x)
#
#         return out.view(input.shape[0], self.N)

class QRDQN(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_support, seed: int=0):
        super(QRDQN, self).__init__()
        self.seed = torch.manual_seed(seed)


        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        # In the second layer the actions will be inserted also
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output layer (single value)
        self.V = nn.Linear(hidden_size[1], num_support)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.V.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.V.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs):
        x = inputs

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # Output
        V = self.V(x)
        return V

# class IQN(nn.Module):
#     def __init__(self, state_size, layer_size, n_cos: int=64, N: int=64, device: str = 'cpu', seed: int=0):
#         super(IQN, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.device = device
#         self.input_shape = state_size
#         self.n_cos = n_cos
#         self.N= N
#         self.layer_size = layer_size
#         self.pis = torch.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(1, 1, self.n_cos).to(self.device)# Starting from 0 as in the paper
#
#         self.head = nn.Linear(self.input_shape, layer_size[0])  # cound be a cnn
#         self.cos_embedding = nn.Linear(self.n_cos, layer_size[0])
#         self.ff_1 = nn.Linear(layer_size[0], layer_size[1])
#         self.ff_2 = nn.Linear(layer_size[1], 1)
#         weight_init([self.cos_embedding, self.head, self.ff_1, self.ff_2])
#
#     def calc_cos(self, batch_size):
#         """
#         Calculating the cosinus values depending on the number of tau samples
#         """
#         taus = torch.rand(batch_size, self.N).to(self.device).unsqueeze(-1)  # (batch_size, n_tau, 1)
#         cos = torch.cos(taus * self.pis).to(self.device)
#
#         assert cos.shape == (batch_size, self.N, self.n_cos), "cos shape is incorrect"
#         return cos, taus
#
#     def forward(self, input):
#         """
#         Quantile Calculation depending on the number of tau
#
#         Return:
#         quantiles [ shape of (batch_size, num_tau, action_size)]
#         taus [shape of ((batch_size, num_tau, 1))]
#
#         """
#         batch_size = input.shape[0]
#
#         x = torch.relu(self.head(input))
#         cos, taus = self.calc_cos(batch_size)  # cos shape (batch, num_tau, layer_size)
#         cos = cos.view(batch_size * self.N, self.n_cos)
#         cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, self.N, self.layer_size[0])  # (batch, n_tau, layer)
#
#         # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
#         x = (x.unsqueeze(1) * cos_x).view(batch_size * self.N, self.layer_size[0])
#
#         x = torch.relu(self.ff_1(x))
#         out = self.ff_2(x)
#         #out1 = out.view(batch_size, self.N)
#         return out.view(batch_size, self.N), taus

# class IQN(nn.Module):
#     def __init__(self, hidden_size, num_inputs, action_dim, num_support, device):
#         super(IQN, self).__init__()
#         num_outputs = action_dim
#         self.num_support = num_support
#         self.device = device
#
#         # Layer 1
#         self.linear1 = nn.Linear(num_inputs, hidden_size[0])
#         self.ln1 = nn.LayerNorm(hidden_size[0])
#
#         # Layer 2
#         # In the second layer the actions will be inserted also
#         self.linear2 = nn.Linear(hidden_size[0] + num_outputs, hidden_size[1])
#         self.ln2 = nn.LayerNorm(hidden_size[1])
#
#         # Output layer (single value)
#         self.V = nn.Linear(hidden_size[1], 1)
#
#         # phi
#         self.phi = nn.Linear(1, hidden_size[1], bias=False)
#         self.phi_bias = nn.Parameter(torch.zeros(hidden_size[1]), requires_grad = True)
#
#         self.linear3 = nn.Linear(hidden_size[1], hidden_size[1])
#
#         # Weight Init
#         fan_in_uniform_init(self.linear1.weight)
#         fan_in_uniform_init(self.linear1.bias)
#
#         fan_in_uniform_init(self.linear2.weight)
#         fan_in_uniform_init(self.linear2.bias)
#
#         nn.init.uniform_(self.V.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
#         nn.init.uniform_(self.V.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)
#
#     def forward(self, inputs, actions):
#         x = inputs
#
#         # Layer 1
#         x = self.linear1(x)
#         x = self.ln1(x)
#         x = F.relu(x)
#
#         # Layer 2
#         x = torch.cat((x, actions), 1)  # Insert the actions
#         x = self.linear2(x)
#         x = self.ln2(x)
#         x = F.relu(x)
#
#         # tau
#         tau = torch.rand(self.num_support, 1)
#         quants = torch.arange(0, self.num_support, 1.0)
#         cos_trans = torch.cos(quants * tau * np.pi).unsqueeze(2) # (num_support, num_support, 1)
#         rand_feat = F.relu(self.phi.to(self.device)(cos_trans).mean(1) + self.phi_bias.to(self.device).unsqueeze(0)).unsqueeze(0)
#
#         x = x.unsqueeze(1).to(self.device)
#         x = x * rand_feat
#
#         x = F.relu(self.linear3.to(self.device)(x))
#
#
#         # Output
#         V = self.V.to(self.device)(x).transpose(1,2) # (bs_size, 1, num_support)
#         V = V.squeeze(1)
#         return V, tau
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class IQN(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_support, device, seed: int=0):
        super(IQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.num_support = num_support

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        # In the second layer the actions will be inserted also
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output layer (single value)
        self.V = nn.Linear(hidden_size[1], 1)

        # phi
        self.phi = nn.Linear(1, hidden_size[1], bias=False)
        self.phi_bias = nn.Parameter(torch.zeros(hidden_size[1]), requires_grad = True)

        self.linear3 = nn.Linear(hidden_size[1], hidden_size[1])

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.V.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.V.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs):
        x = inputs

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # tau
        tau = torch.rand(self.num_support, 1).to(device)
        quants = torch.arange(0, self.num_support, 1.0).to(device)
        cos_trans = torch.cos(quants * tau * np.pi).unsqueeze(2) # (num_support, num_support, 1)
        rand_feat = F.relu(self.phi(cos_trans).mean(1) + self.phi_bias.unsqueeze(0)).unsqueeze(0)

        x = x.unsqueeze(1)
        x = x * rand_feat

        x = F.relu(self.linear3(x))


        # Output
        V = self.V(x).transpose(1,2) # (bs_size, 1, num_support)
        V = V.squeeze(1)
        return V, tau


class SplineDQN(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_support, device, seed:int=0):
        super(SplineDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = device

        self.num_support = num_support
        self.K = num_support

        self.min_bin_width = 1e-3
        self.min_bin_height = 1e-3
        self.min_derivative = 1e-3

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        # In the second layer the actions will be inserted also
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output layer (knots)
        self.V = nn.Linear(hidden_size[1], (3 * self.K - 1))

        # Scale
        self.alpha = nn.Linear(hidden_size[1], 1)
        self.beta = nn.Linear(hidden_size[1], 1)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        fan_in_uniform_init(self.V.weight)
        fan_in_uniform_init(self.V.bias)

        nn.init.uniform_(self.alpha.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.alpha.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

        nn.init.uniform_(self.beta.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.beta.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        x = inputs

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # Output
        spline_param = self.V(x)
        scale_a = self.alpha(x)
        scale_a = torch.exp(scale_a)
        scale_b = self.beta(x)

        # split the last dimention to W, H, D
        W, H, D = torch.split(spline_param, self.K, dim=1)
        W, H = torch.softmax(W, dim=1), torch.softmax(H, dim=1)
        W = self.min_bin_width + (1 - self.min_bin_width * self.K) * W
        H = self.min_bin_height + (1 - self.min_bin_height * self.K) * H
        D = self.min_derivative + F.softplus(D)
        D = F.pad(D, pad=(1, 1))
        constant = np.log(np.exp(1 - 1e-3) - 1)
        D[..., 0] = constant
        D[..., -1] = constant

        # start and end x of each bin
        cumwidths = torch.cumsum(W, dim=-1).to(self.device)
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
        cumwidths[..., -1] = 1.0
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]  # (batch_sz, K)

        # start and end y of each bin
        cumheights = torch.cumsum(H, dim=-1)
        cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
        cumheights = (scale_a * cumheights + scale_b).to(self.device)
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        # get bin index for each tau
        tau = torch.arange(0.5 * (1 / self.num_support), 1, 1 / self.num_support).to(self.device)
        tau = tau.expand((batch_size, self.num_support))

        cumwidths_expand = cumwidths.unsqueeze(dim=1).to(self.device)
        cumwidths_expand = cumwidths_expand.expand(-1, self.num_support, -1)  # (batch_sz, num_support, K+1)

        bin_idx = self.searchsorted_(cumwidths_expand, tau)

        # collect number
        input_cumwidths = cumwidths.gather(-1, bin_idx)
        input_bin_widths = widths.gather(-1, bin_idx)

        input_cumheights = cumheights.gather(-1, bin_idx)
        input_heights = heights.gather(-1, bin_idx)

        delta = heights / widths

        input_delta = delta.gather(-1, bin_idx)

        input_derivatives = D.to(self.device).gather(-1, bin_idx)
        input_derivatives_plus_one = D[..., 1:].to(self.device).gather(-1, bin_idx)

        # calculate quadratic spline for each tau
        theta = (tau - input_cumwidths) / input_bin_widths

        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + (
                    input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        outputs = input_cumheights + numerator / denominator

        return outputs

    def searchsorted_(self, bin_locations, inputs):
        return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1

class NCQR(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_support, device, seed:int=0):
        super(NCQR, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device=device

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        # In the second layer the actions will be inserted also
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output layer (single value)
        self.V = nn.Linear(hidden_size[1], num_support)

        # Scale
        self.alpha = nn.Linear(hidden_size[1], 1)
        self.beta = nn.Linear(hidden_size[1], 1)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        fan_in_uniform_init(self.V.weight)
        fan_in_uniform_init(self.V.bias)

        nn.init.uniform_(self.alpha.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.alpha.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

        nn.init.uniform_(self.beta.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.beta.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs):
        x = inputs

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # Output
        quant = self.V(x)
        quant = torch.softmax(quant, dim=-1)
        quant = torch.cumsum(quant, dim=-1)

        # scale
        scale_a = F.relu(self.alpha(x))
        scale_b = self.beta(x)

        V = scale_a * quant + scale_b
        return V