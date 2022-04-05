## Champsim Modified for Trace Generation
This is a modified version of champsim used to extract info from traces and write them in to text files in CSV format

## Changelog
- Added a trace recorder source file for brach and cache replacemnent neamed trace.cc.
- For branch this uses the bimodal predictor as the base and for cache it uses lru.
- Configured Champsim to use these file in my_config.json.
- Updated the function definitions for generate inc files and APIs in the config.sh script and in the cache.cc to add extrad fields (origin field added).
- Built champsim with these changes - using make.
- Added a line to reduce LLC cache size to my_config.json - sets reduced to 512 from 2048 (16MB to 4MB).
