关于MDAnanlysis

ts对象：timestep 就是每一个时间步（每一个帧）

A `Timestep` represents all data for a given frame in a trajectory.

The data inside a `Timestep` is often accessed indirectly through a [`AtomGroup`](https://docs.mdanalysis.org/stable/documentation_pages/core/groups.html#MDAnalysis.core.groups.AtomGroup) but it is also possible to manipulate Timesteps directly.

我这儿要干的事情就是直接处理timesteps

直接使用ts.positions会丢失原子信息。