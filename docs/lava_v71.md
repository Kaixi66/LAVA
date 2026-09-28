# V7.1: paired positive reuse

V7 representation is unchanged: unconditioned state-delta world encoder,
8x16 queries, depth-2 LogSignature, time channel on, graded-soft rho=1,
negative squared L2, final policy hidden, temperature .07, lambda .01.

With batch size 128 and sample ratio .125, exactly 16 LAVA positives form
8 pairs from distinct physical episodes. Select a task uniformly among
remaining eligible tasks, then an episode uniformly without replacement.
For the batch's common scale L, choose two legal interval starts separated
by at least 2L. Episodes shorter than 3L+1 cannot supply a pair and are
excluded at that scale. Choose a valid relative action-chunk offset for
each interval; load each segment's own observation, actions and future target.
The other 112 samples follow the existing shuffled ordinary sample stream.
All 128 samples receive normal policy/future supervision. This changes
the composition of the base batch too, as required by paired positives.

Each row scores exactly its own positive, its unique episode partner, and
four uniformly selected distinct same-scale positive paths from other
episodes. Same-task other episodes are allowed. Cross candidates may include
both paths of another episode. There is no weighting, family averaging,
order perturbation, or explicit-negative RGB. Candidates reuse differentiable
positive signatures; both sides receive gradients. Production batches have
14 available cross paths, so every row has six finite logits. The helper
supports fewer cross paths by masking missing entries, but never drops a
missing/invalid temporal partner silently.

The sampler maintains the existing batch-uniform scale cycle and checkpoint
RNG state. Its positive sampling is now explicit task/episode/pair sampling,
not the old independent per-frame Bernoulli selection. Dataset statistics
describing the old Bernoulli probabilities are not the new sampler's quotas.
Actual sampling counts and scale fields remain logged.

Monitoring adds Paired_Hard_Count, Cross_Count, Hard_Acc, Hard_Margin,
Hard_Probability, Cross_Probability, Positive_Probability, Candidate_Acc,
World_Path_Count and Distance_Over_L. Combine these with Scale_Min/Max,
branch gradients, policy/LAVA gradient cosine, losses and update time.
Probability mass is not a parameter-gradient fraction; temporal separation
does not guarantee semantic negativity or that a candidate is difficult.

Unit tests verify the sampler, restore, dataset/collation, six-logit CE,
single world encoding, all scales and action/world autograd. The NVL job
first runs ten full-batch GPU updates at full LAVA weight in a separate
no-checkpoint preflight, then initializes the 12-epoch experiment afresh.
Its dependent L40S evaluation runs the final epoch on 10 demo_clean tasks,
50 rollouts each, evaluation seed 0.
