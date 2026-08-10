#Manual{,
#title = {nflreadr: Download 'nflverse' Data},
#author = {Tan Ho and Sebastian Carl},
#note = {R package version 1.5.1.9000},
#url = {https://nflreadr.nflverse.com},
#}

#Runs the command to obtain the libraries from NFLverse
install.packages("nflverse",
  repos = c("https://nflverse.r-universe.dev", getOption("repos"))
)

#Selects the libraries where I want to retrieve the data
library(nflreadr)
library(dplyr)

#Loads the play by play data from 2020 to 2023
pbp <- load_pbp(2020:2023)

#Creates nflverse_game_id so the left_join statement works
pbp <- pbp %>%
  mutate(nflverse_game_id = game_id)

#Loads the participation data (contains personnel)
part <- load_participation(2020:2023)

#Merges participation onto pbp using the game id and play id
pbp2 <- pbp %>%
  left_join(part, by = c("nflverse_game_id", "play_id"))

#Selects the features I want for the model
pbp_export <- pbp2 |>
  select(
    play_id, game_id, season,
    play_type, down, ydstogo, yardline_100,
    score_differential, game_seconds_remaining,
    offense_personnel, defense_personnel,
    posteam
  )

#Exports the selected data to a csv file
write.csv(pbp_export, "data/pbp_export.csv", row.names = FALSE)
