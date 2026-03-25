import collections
import time

class TemporalVoter:
    def __init__(self, window_size=10, majority_threshold=7, cooldown=10):
        """
        Maintains a sliding window of recent detections for Majority Voting.
        
        Args:
            window_size (int): Size of the sliding window.
            majority_threshold (int): Minimum count of same ID in window for confirmation.
            cooldown (int): Seconds before the same person can be confirmed again.
        """
        self.window_size = window_size
        self.majority_threshold = majority_threshold
        self.cooldown_seconds = cooldown
        
        self.buffer = collections.deque(maxlen=window_size)
        self.last_confirmed_id = None
        self.last_confirmed_time = 0
        
    def vote(self, user_id):
        """
        Add a new detection to the window and check for majority consensus.
        
        Returns:
            tuple: (is_confirmed, voted_id)
        """
        self.buffer.append(user_id)
        
        if len(self.buffer) < self.window_size:
            return False, user_id
            
        # Count occurrences in window (excluding "Unknown")
        counts = collections.Counter([x for x in self.buffer if x != "Unknown"])
        if not counts:
            return False, "Unknown"
            
        best_id, count = counts.most_common(1)[0]
        
        if count >= self.majority_threshold:
            # Check cooldown to prevent duplicate attendance logs
            now = time.time()
            if best_id == self.last_confirmed_id and (now - self.last_confirmed_time < self.cooldown_seconds):
                return False, best_id
                
            self.last_confirmed_id = best_id
            self.last_confirmed_time = now
            # Clear buffer after confirmation to prevent "double-triggering" next frame
            self.buffer.clear()
            return True, best_id
            
        return False, best_id

    def reset(self):
        """Reset the buffer."""
        self.buffer.clear()
