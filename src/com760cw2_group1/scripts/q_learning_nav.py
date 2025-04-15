#!/usr/bin/env python

import rospy
import numpy as np
import random
import math
import time
from collections import deque
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from com760cw2_group1.msg import State, Action, Reward
from com760cw2_group1.srv import LearnNavigation, LearnNavigationResponse

class QLearningNav:
    def __init__(self):
        rospy.init_node('q_learning_nav', anonymous=True)
        # Learning parameters - adjusted for better stability
        self.alpha = 0.1  # Reduced learning rate for more stable learning
        self.gamma = 0.95  # Slightly increased discount factor to value future rewards more
        self.epsilon = 0.5  # Higher starting epsilon for better exploration
        self.epsilon_decay = 0.998  # Slower decay for more exploration
        self.epsilon_min = 0.05  # Slightly higher minimum for continued exploration
        
        # Enhanced state representation
        self.num_laser_readings = 9  # Increased from 5 for better spatial awareness
        self.num_distance_bins = 4  # Increased from 3 for finer granularity
        self.num_angle_bins = 12  # Increased from 8 for better directional awareness
        
        # Actions remain the same for now
        self.actions = [0, 1, 2, 3]  # forward, left, right, stop
        self.num_actions = len(self.actions)
        
        # Initialize Q-table with new state space dimensions
        self.q_table = {}  # Using dictionary for sparse representation instead of full array
        
        # Position tracking
        self.position = [0, 0]
        self.orientation = 0
        self.goal = [5, 5]
        self.start_positions = [
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],  # Add various starting positions
            [2, 2], [-2, -2], [3, -3], [-3, 3]
        ]
        
        # Publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher('/Group1Bot/cmd_vel', Twist, queue_size=10)
        self.laser_sub = rospy.Subscriber('/Group1Bot/laser/scan', LaserScan, self.laser_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.state_pub = rospy.Publisher('/Group1Bot/state', State, queue_size=10)
        self.action_pub = rospy.Publisher('/Group1Bot/action', Action, queue_size=10)
        self.reward_pub = rospy.Publisher('/Group1Bot/reward', Reward, queue_size=10)
        self.learn_service = rospy.Service('/Group1Bot/learn_navigation', LearnNavigation, self.learn_navigation_callback)
        
        # State tracking
        self.laser_data = None
        self.current_state = None
        self.current_discrete_state = None
        self.previous_distance_to_goal = None
        self.previous_action = None
        self.collision = False
        self.last_action_time = rospy.get_time()
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        
        # Goal zone parameters
        self.goal_reached_distance = 0.5
        
        # Tracking metrics
        self.episode_steps = []
        self.episode_rewards = []
        
        rospy.loginfo("Enhanced Q-Learning Navigation initialized!")

    def laser_callback(self, data):
        self.laser_data = data.ranges
        # Check for collision based on minimum distance
        min_distance = min([x for x in data.ranges if not math.isinf(x)], default=float('inf'))
        self.collision = min_distance < 0.3
        if self.collision:
            rospy.logwarn_throttle(1.0, "Collision detected! Min distance: %.2f", min_distance)

    def odom_callback(self, data):
        self.position[0] = data.pose.pose.position.x
        self.position[1] = data.pose.pose.position.y
        quaternion = (
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w
        )
        _, _, self.orientation = euler_from_quaternion(quaternion)

    def get_state(self):
        if self.laser_data is None:
            rospy.logwarn_throttle(1.0, "No laser data available")
            return None
            
        # Calculate distance and angle to goal
        dx = self.goal[0] - self.position[0]
        dy = self.goal[1] - self.position[1]
        distance_to_goal = math.sqrt(dx*dx + dy*dy)
        angle_to_goal = math.atan2(dy, dx) - self.orientation
        
        # Normalize angle to [-pi, pi]
        if angle_to_goal > math.pi:
            angle_to_goal -= 2 * math.pi
        elif angle_to_goal < -math.pi:
            angle_to_goal += 2 * math.pi
            
        # Process laser data to extract minimum distances in sectors
        simplified_laser = []
        sector_size = len(self.laser_data) // self.num_laser_readings
        for i in range(self.num_laser_readings):
            start_idx = i * sector_size
            end_idx = (i + 1) * sector_size
            sector_data = self.laser_data[start_idx:end_idx]
            # Replace inf with a max distance and find minimum in each sector
            sector_data = [min(x, 3.5) if not math.isinf(x) else 3.5 for x in sector_data]
            simplified_laser.append(min(sector_data))  # Use minimum instead of average
            
        # Publish state for visualization/debugging
        state_msg = State()
        state_msg.laser_readings = simplified_laser
        state_msg.goal_distance = distance_to_goal
        state_msg.goal_angle = angle_to_goal
        self.state_pub.publish(state_msg)
        
        # Discretize state components
        discrete_laser = tuple(self.discretize_laser(reading) for reading in simplified_laser)
        discrete_distance = self.discretize_distance(distance_to_goal)
        discrete_angle = self.discretize_angle(angle_to_goal)
        
        # Store raw state for reward calculation
        self.current_state = {
            'laser': simplified_laser,
            'distance': distance_to_goal,
            'angle': angle_to_goal,
            'position': self.position.copy(),
            'orientation': self.orientation
        }
        
        # Create discrete state tuple (immutable for dictionary key)
        self.current_discrete_state = (
            discrete_laser,
            discrete_distance,
            discrete_angle
        )
        
        # Store previous distance for reward calculation
        if self.previous_distance_to_goal is None:
            self.previous_distance_to_goal = distance_to_goal
            
        rospy.logdebug("State: distance=%.2f, angle=%.2f", distance_to_goal, angle_to_goal)
        return self.current_discrete_state

    def discretize_laser(self, reading):
        # More bins for laser readings
        if reading < 0.3:
            return 0  # Very close - danger
        elif reading < 0.6:
            return 1  # Close
        elif reading < 1.0:
            return 2  # Medium
        elif reading < 2.0:
            return 3  # Far
        else:
            return 4  # Very far

    def discretize_distance(self, distance):
        # More bins for distance to goal
        if distance < self.goal_reached_distance:
            return 0  # Goal reached
        elif distance < 1.0:
            return 1  # Very close to goal
        elif distance < 2.0:
            return 2  # Close to goal
        elif distance < 4.0:
            return 3  # Medium distance
        else:
            return 4  # Far from goal

    def discretize_angle(self, angle):
        # More bins for angle to goal
        normalized_angle = angle + math.pi
        sector = int(normalized_angle / (2 * math.pi / self.num_angle_bins))
        return min(sector, self.num_angle_bins - 1)

    def get_q_value(self, state, action):
        # Get Q-value from sparse dictionary or return 0.0 if not found
        if state is None:
            return 0.0
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
            
        return self.q_table[state][action]

    def choose_action(self, state):
        if state is None:
            action = random.choice(self.actions)
        else:
            # Epsilon-greedy with decay
            if random.random() < self.epsilon:
                action = random.choice(self.actions)
            else:
                # Get Q-values for this state
                if state not in self.q_table:
                    self.q_table[state] = np.zeros(self.num_actions)
                    
                # Handle equal Q-values by random selection among best actions
                q_values = self.q_table[state]
                max_q = np.max(q_values)
                best_actions = [a for a, q in enumerate(q_values) if q == max_q]
                action = random.choice(best_actions)
        
        # Publish action for monitoring
        action_msg = Action()
        action_msg.action_id = action
        self.action_pub.publish(action_msg)
        
        self.previous_action = action
        self.last_action_time = rospy.get_time()
        
        action_names = ["forward", "left", "right", "stop"]
        rospy.loginfo("Chose action: %d (%s)", action, action_names[action])
        return action

    def take_action(self, action):
        twist = Twist()
        
        # Adjusted velocities for smoother motion
        if action == 0:  # Move forward
            twist.linear.x = 0.3  # Reduced speed for safety
            twist.angular.z = 0.0
        elif action == 1:  # Turn left
            twist.linear.x = 0.1  # Keep some forward motion during turns
            twist.angular.z = 0.5  # Reduced for smoother turns
        elif action == 2:  # Turn right
            twist.linear.x = 0.1
            twist.angular.z = -0.5
        elif action == 3:  # Stop
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            
        self.cmd_vel_pub.publish(twist)
        
        # Wait for action to have effect - adaptive timing based on action
        wait_time = 0.2 if action == 3 else 0.3  # Wait longer for movement actions
        rospy.sleep(wait_time)
        
        rospy.logdebug("Published cmd_vel: linear.x=%.2f, angular.z=%.2f", twist.linear.x, twist.angular.z)

    def get_reward(self):
        if self.current_state is None or self.previous_distance_to_goal is None:
            return 0, False
            
        # Extract current state information
        current_distance = self.current_state['distance']
        current_angle = self.current_state['angle']
        
        # Default values
        reward = 0
        done = False
        
        # Check terminal conditions first
        if self.collision:
            reward = -20  # Increased penalty for collisions
            done = True
            rospy.logwarn("Collision detected! Reward: %.2f", reward)
        elif current_distance < self.goal_reached_distance:
            # Bonus for reaching goal and higher reward for faster completion
            reward = 50
            done = True
            rospy.loginfo("Goal reached! Reward: %.2f", reward)
        else:
            # Progress reward based on distance change
            distance_change = self.previous_distance_to_goal - current_distance
            progress_reward = distance_change * 10  # Increased multiplier
            
            # Heading reward - encourage facing the goal
            angle_factor = 1.0 - min(abs(current_angle) / math.pi, 1.0)
            heading_reward = angle_factor * 0.5
            
            # Time penalty to encourage efficiency
            time_penalty = -0.1
            
            # Action-specific penalties to discourage excessive stopping
            action_penalty = -0.5 if self.previous_action == 3 else 0
            
            # Calculate proximity penalty (being too close to obstacles)
            proximity_penalty = 0
            if self.laser_data:
                min_dist = min([x for x in self.laser_data if not math.isinf(x)], default=float('inf'))
                if min_dist < 0.5 and min_dist > 0.3:  # Close but not collision
                    proximity_penalty = -0.3 * (0.5 - min_dist) / 0.2
            
            # Sum all reward components
            reward = progress_reward + heading_reward + time_penalty + action_penalty + proximity_penalty
            
            # Cap reward to reasonable range
            reward = max(min(reward, 5), -5)
            
        # Update previous distance
        self.previous_distance_to_goal = current_distance
        
        # Publish reward for monitoring
        reward_msg = Reward()
        reward_msg.reward_value = reward
        reward_msg.episode_done = done
        self.reward_pub.publish(reward_msg)
        
        return reward, done

    def update_q_table(self, state, action, reward, next_state, done):
        if state is None or (next_state is None and not done):
            return
            
        # Initialize Q-values if not present
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
            
        if not done and next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_actions)
            
        # Update Q-value
        if done:
            # Terminal state update
            self.q_table[state][action] += self.alpha * (reward - self.q_table[state][action])
        else:
            # Double Q-learning approach to reduce overestimation
            best_action = np.argmax(self.q_table[next_state])
            max_future_q = self.q_table[next_state][best_action]
            
            current_q = self.q_table[state][action]
            new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
            self.q_table[state][action] = new_q

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        if state is not None:  # Only add valid experiences
            self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_from_replay_buffer(self):
        """Sample a batch of experiences from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return self.replay_buffer  # Return all if buffer smaller than batch size
        return random.sample(self.replay_buffer, self.batch_size)

    def learn_from_replay(self):
        """Learn from a batch of experiences"""
        if len(self.replay_buffer) < 10:  # Minimum buffer size before learning
            return
            
        batch = self.sample_from_replay_buffer()
        for state, action, reward, next_state, done in batch:
            self.update_q_table(state, action, reward, next_state, done)

    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def learn_navigation_callback(self, req):
        """Service callback to start learning"""
        if req.start_learning:
            success = self.run_q_learning(req.max_episodes, req.max_steps_per_episode)
            if success:
                return LearnNavigationResponse(True, "Q-Learning navigation completed successfully")
            else:
                return LearnNavigationResponse(False, "Q-Learning navigation failed")
        return LearnNavigationResponse(False, "Learning not started")

    def reset_episode(self):
        """Reset environment for new episode"""
        # Stop the robot
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(0.5)
        
        # Reset state variables
        self.collision = False
        self.previous_distance_to_goal = None
        
        # TODO: In a real simulation, you would teleport the robot to a new start position
        # For now, we'll just pretend and update the internal position tracking
        # In a real implementation, this should make service calls to the simulator
        start_pos = random.choice(self.start_positions)
        rospy.loginfo(f"Starting new episode at position {start_pos}")
        
        # Wait for laser data to ensure valid state
        start_time = rospy.get_time()
        while self.laser_data is None and not rospy.is_shutdown() and (rospy.get_time() - start_time) < 5.0:
            rospy.logwarn_throttle(1.0, "Waiting for laser data in reset_episode")
            rospy.sleep(0.1)
            
        return self.get_state()

    def run_q_learning(self, max_episodes=200, max_steps=300):
        """Run Q-learning algorithm"""
        try:
            rospy.loginfo("Starting Enhanced Q-Learning with %d episodes...", max_episodes)
            total_steps = 0
            
            # Initialize statistics tracking
            self.episode_steps = []
            self.episode_rewards = []
            
            for episode in range(max_episodes):
                state = self.reset_episode()
                if state is None:
                    rospy.logerr("Failed to get initial state, skipping episode %d", episode+1)
                    continue
                    
                episode_reward = 0
                step = 0
                
                for step in range(max_steps):
                    # Choose and take action
                    action = self.choose_action(state)
                    self.take_action(action)
                    
                    # Get next state and reward
                    next_state = self.get_state()
                    reward, done = self.get_reward()
                    episode_reward += reward
                    
                    # Store experience and update Q-values
                    self.add_to_replay_buffer(state, action, reward, next_state, done)
                    self.update_q_table(state, action, reward, next_state, done)
                    
                    # Learn from past experiences
                    if (step + 1) % 5 == 0:  # Learn every 5 steps
                        self.learn_from_replay()
                        
                    # Move to next state
                    state = next_state
                    total_steps += 1
                    
                    if done:
                        break
                        
                # Decay exploration rate
                self.decay_epsilon()
                
                # Track episode statistics
                self.episode_steps.append(step + 1)
                self.episode_rewards.append(episode_reward)
                
                # Print progress
                avg_steps = sum(self.episode_steps[-10:]) / min(10, len(self.episode_steps))
                avg_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
                
                rospy.loginfo("Episode: %d/%d, Steps: %d, Reward: %.2f, Eps: %.4f, Avg10-Steps: %.1f, Avg10-Reward: %.2f",
                              episode+1, max_episodes, step+1, episode_reward, self.epsilon, avg_steps, avg_reward)
                              
                # Save Q-table periodically
                if (episode + 1) % 20 == 0:
                    self.save_q_table(f"q_table_episode_{episode+1}.pkl")
                    
            # Save final Q-table
            self.save_q_table("q_table_final.pkl")
            rospy.loginfo("Enhanced Q-Learning completed after %d total steps!", total_steps)
            return True
            
        except Exception as e:
            rospy.logerr("Error during Q-Learning: %s", e)
            import traceback
            rospy.logerr(traceback.format_exc())
            return False

    def save_q_table(self, filename):
        """Save Q-table to file"""
        try:
            import pickle
            with open(filename, 'wb') as f:
                pickle.dump(self.q_table, f)
            rospy.loginfo("Q-table saved to %s", filename)
        except Exception as e:
            rospy.logerr("Error saving Q-table: %s", e)

    def load_q_table(self, filename):
        """Load Q-table from file"""
        try:
            import pickle
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            rospy.loginfo("Q-table loaded from %s with %d state entries", filename, len(self.q_table))
            return True
        except Exception as e:
            rospy.logerr("Error loading Q-table: %s", e)
            return False
    
    def run_evaluation(self, num_episodes=10, max_steps=300):
        """Run evaluation using learned policy"""
        rospy.loginfo("Starting evaluation with %d episodes...", num_episodes)
        success_count = 0
        total_rewards = 0
        total_steps = 0
        
        # Save original epsilon and set to minimum for evaluation
        original_epsilon = self.epsilon
        self.epsilon = 0.01  # Small epsilon for minimal exploration
        
        try:
            for episode in range(num_episodes):
                state = self.reset_episode()
                if state is None:
                    rospy.logerr("Failed to get initial state, skipping evaluation episode %d", episode+1)
                    continue
                    
                episode_reward = 0
                step = 0
                
                for step in range(max_steps):
                    action = self.choose_action(state)
                    self.take_action(action)
                    next_state = self.get_state()
                    reward, done = self.get_reward()
                    episode_reward += reward
                    state = next_state
                    
                    if done:
                        if reward > 0:  # Positive reward means goal reached
                            success_count += 1
                        break
                        
                total_rewards += episode_reward
                total_steps += step + 1
                
                rospy.loginfo("Eval Episode %d/%d: Steps=%d, Reward=%.2f, Success=%s", 
                              episode+1, num_episodes, step+1, episode_reward, reward > 0)
                              
            # Restore original epsilon
            self.epsilon = original_epsilon
            
            avg_reward = total_rewards / num_episodes
            avg_steps = total_steps / num_episodes
            success_rate = success_count / num_episodes * 100
            
            rospy.loginfo("Evaluation Results: Success Rate=%.1f%%, Avg Reward=%.2f, Avg Steps=%.1f",
                          success_rate, avg_reward, avg_steps)
                          
            return success_rate
            
        except Exception as e:
            rospy.logerr("Error during evaluation: %s", e)
            # Restore original epsilon
            self.epsilon = original_epsilon
            return 0.0

if __name__ == '__main__':
    try:
        q_learning_nav = QLearningNav()
        
        # Check if we should load a saved Q-table
        import sys
        if len(sys.argv) > 1:
            q_table_file = sys.argv[1]
            if q_learning_nav.load_q_table(q_table_file):
                # Run evaluation if Q-table was loaded
                q_learning_nav.run_evaluation(num_episodes=5)
        
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
