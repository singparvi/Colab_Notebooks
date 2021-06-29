


class User:
    def __init__(self, mod_status, username):
        self.mod_status = mod_status
        self.username = username
        self.reputation = 0
        self.banned = False


    def upvote_thread(self, thread, time_voted):
        vote = Vote(up=True, time_voted=time_voted, voter=self.username, thread=thread)
        return vote


class Moderator(User):
    def __init__(self, mod_level, mod_subreddits, username):
        self.mod_level = mod_level
        self.mod_subreddits = mod_subreddits
        User.__init__(self,mod_status=True, username=username)


    def ban_user(self, user):
        user.banned = True


class Vote:
    def __init__(self, up, time_voted, voter, thread):
        self.up = up
        self.time_voted = time_voted
        self.voter = voter
        self.thread = thread
        self.is_deleted = False

    def delete(self):
        self.is_deleted = True


class Thread:
    def __init__(self, title, parent_thread):
        self.title = title
        self.parent_thread = parent_thread
        self.upvote_count = 0
        self.downvote_count = 0
        self.comments = []
        self.is_featured = False

    def feature_thread(self):
        self.is_featured = True





sarah = User(mod_status=False, username='sarah119')
moto_thread = Thread('motorcycles are cool', 'things that are cool')
sarahs_vote = sarah.upvote_thread(moto_thread, 'now')
print(sarahs_vote.is_deleted)
sarahs_vote.delete()
print(sarahs_vote.is_deleted)
jill = Moderator(mod_level=99, mod_subreddits=['motorcycles'], username='modhill')
jill.ban_user(sarah)