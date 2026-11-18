# Evaluator trips go here (Laptop 1 only)

Copy each of the 10 reserved evaluation trips here as its own folder:

```
data/eval/<trip_id>/<trip_id>.track
data/eval/<trip_id>/<trip_id>.nodes
data/eval/<trip_id>/<trip_id>.arcs
data/eval/<trip_id>/<trip_id>.route   # read only by the server-side scorer
```

This folder is git-ignored. No training, tuning or analysis script reads it.
