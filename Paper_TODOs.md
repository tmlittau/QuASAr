# TODOs for the Paper

## 1. Benchmarking
- make theoretical analysis plots from the partitioning_theory notebook "nicer" (using seaborn style and better coloring)
- improve current included single-method-plots optically
- add more large-scale benchmarks based on the analyses in the partitioning notebooks

## 2. Explanatory content
- Add explanation for the different conversion-optimizations
    - **B2B**: boundary-to-boundary for MPS conversion of specific partitions
    - **LW**: local-window extraction for small dense window of qubits before conversion
    - **ST**: staged conversion through "capped" representation with bond dimension
    - **full**: a full state extraction, used when all other options are deemed infeasible
- Accuracy Estimation for complete MPS cost estimation

## 3. Rihan Suggestions
- Intro
    - too high level and general
    - be more explicit with Research Problem
    - existing sims cannot choose the repr well
    - we borrow cost model from DBMS
    - more speedup through system design
    - build everything on the optimizations and system architecture
    - challenge has to be very concrete and formalised research problem
    - for contributions (1.1) three or four items and last one is always results (speedup etc.)
- Background
    - not easy to understand
    - should be like a mini-tutorial, prepare example for it
    - write the repr's as if I am defining them
    - a bit too long, should be more compressed
- Architecture
    - all things that are optimizations put them in sec 4
    - put in the figure of the architecture
- Section 4: Optimisations
- Section 5
    - better call it Cost Models/Cost Estimation 
- Section 6: 
    - compress and put in appendix
    - the backend adapters should be as representations and with framework and resp. versions in the text
    - This way baselines are not too confusing in relation to this
- Evaluations
    - each subsection title should correspond to research questions
    - if I don't have a certain optimization what is the result without it (justifying it's implementation)
    - ablation studies
- Related Works after Evaluations