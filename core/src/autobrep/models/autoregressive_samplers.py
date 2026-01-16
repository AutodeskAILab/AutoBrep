from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field
from x_transformers.autoregressive_wrapper import min_p, top_a, top_k, top_p


class TopP(BaseModel):
    """
    Nucleus sampling.

    Top-p sampling sets a threshold (`top_p_threshold`) and restricts the sampling to
    the set of most probable tokens with cumulative probability more than p.
    Probabilities of the token from this set are then rescaled to sum to 1 and the
    rest are rejected.

    Proposed by Ari Holtzman et al. in 2019.
    """

    name: Literal["top_p"] = "top_p"
    top_p_threshold: Annotated[
        float,
        Field(
            description="The cumulative probability threshold below which tokens are rejected.",
            gt=0,
            le=1,
        ),
    ] = 0.8

    def __call__(self, logits):
        return top_p(logits, self.top_p_threshold)


class TopK(BaseModel):
    """
    Top-k sampling restricts the sampling to the set of k most probable tokens.
    Probabilities of the token from this set are then rescaled to sum to 1 and the
    rest are rejected.
    """

    name: Literal["top_k"] = "top_k"
    frac_num_tokens: Annotated[
        float,
        Field(
            description="The fraction of the vocabulary to not be rejected."
            " If `num_sampled_token` is provided this is ignored.",
            ge=0.0,
            le=1.0,
        ),
    ] = 0.1
    num_sampled_tokens: Annotated[
        Optional[int],
        Field(
            description="The total number of tokens to not be rejected."
            " This takes priority over `frac_num_tokens`.",
            gt=1,
        ),
    ] = None

    def __call__(self, logits):
        return top_k(logits, self.frac_num_tokens, self.num_sampled_tokens)


class TopA(BaseModel):
    """
    Top-a sampling restricts the sampling to the set of tokens with the ratio
    of probability to the maximum probability raised to a power greater
    than a threshold: p / (p_max ^ p_pow) >= p_ratio.
    Probabilities of the token from this set are then rescaled to sum to 1 and the
    rest are rejected.
    """

    name: Literal["top_a"] = "top_a"
    p_pow: Annotated[
        float, Field(description="The power to raise the maximum probability to.", gt=1)
    ] = 2.0
    p_ratio: Annotated[
        float,
        Field(
            description="The ratio threshold to reject tokens.",
            gt=0.0,
            le=1.0,
        ),
    ] = 0.02

    def __call__(self, logits):
        return top_a(logits, self.p_pow, self.p_ratio)


class MinP(BaseModel):
    """
    Min-p sampling restricts the sampling to the set of tokens with probability
    greater than the maximum probability times a constant.
    Probabilities of the token from this set are then rescaled to sum to 1 and the
    rest are rejected.
    https://arxiv.org/abs/2407.01082
    """

    name: Literal["min_p"] = "min_p"
    min_p: Annotated[
        float,
        Field(
            description="The constant by which the maximum probability is multiplied to form the threshold.",
            ge=0,
            le=1,
        ),
    ] = 0.1

    def __call__(self, logits):
        return min_p(logits, self.min_p)
