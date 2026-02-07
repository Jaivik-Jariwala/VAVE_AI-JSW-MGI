-- SQL command to add current_scenario_image and proposal_scenario_image columns to the ideas table
-- Run this command on your PostgreSQL database

ALTER TABLE ideas 
ADD COLUMN IF NOT EXISTS current_scenario_image TEXT,
ADD COLUMN IF NOT EXISTS proposal_scenario_image TEXT;

-- Optional: Add comments to document the columns
COMMENT ON COLUMN ideas.current_scenario_image IS 'Path to the current scenario image for the idea';
COMMENT ON COLUMN ideas.proposal_scenario_image IS 'Path to the proposal scenario image for the idea';

