package data

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"strconv"
	"testing"
	"time"
)

var positiveFeedbackType = "positiveFeedbackType"
var negativeFeedbackType = "negativeFeedbackType"

func getUsers(t *testing.T, db Database) []User {
	users := make([]User, 0)
	var err error
	var data []User
	cursor := ""
	for {
		cursor, data, err = db.GetUsers(cursor, 2)
		assert.Nil(t, err)
		users = append(users, data...)
		if cursor == "" {
			if _, ok := db.(*Redis); !ok {
				assert.LessOrEqual(t, len(data), 2)
			}
			return users
		} else {
			if _, ok := db.(*Redis); !ok {
				assert.Equal(t, 2, len(data))
			}
		}
	}
}

func getItems(t *testing.T, db Database) []Item {
	items := make([]Item, 0)
	var err error
	var data []Item
	cursor := ""
	for {
		cursor, data, err = db.GetItems(cursor, 2)
		assert.Nil(t, err)
		items = append(items, data...)
		if cursor == "" {
			if _, ok := db.(*Redis); !ok {
				assert.LessOrEqual(t, len(data), 2)
			}
			return items
		} else {
			if _, ok := db.(*Redis); !ok {
				assert.Equal(t, 2, len(data))
			}
		}
	}
}

func getFeedback(t *testing.T, db Database, feedbackType *string) []Feedback {
	feedback := make([]Feedback, 0)
	var err error
	var data []Feedback
	cursor := ""
	for {
		cursor, data, err = db.GetFeedback(cursor, 2, feedbackType)
		assert.Nil(t, err)
		feedback = append(feedback, data...)
		if cursor == "" {
			if _, ok := db.(*Redis); !ok {
				assert.LessOrEqual(t, len(data), 2)
			}
			return feedback
		} else {
			if _, ok := db.(*Redis); !ok {
				assert.Equal(t, 2, len(data))
			}
		}
	}
}

func testUsers(t *testing.T, db Database) {
	// Insert users
	for i := 9; i >= 0; i-- {
		if err := db.InsertUser(User{
			UserId:  strconv.Itoa(i),
			Labels:  []string{strconv.Itoa(i + 100)},
			Comment: fmt.Sprintf("comment %d", i),
		}); err != nil {
			t.Fatal(err)
		}
	}
	// Get users
	users := getUsers(t, db)
	assert.Equal(t, 10, len(users))
	for i, user := range users {
		assert.Equal(t, strconv.Itoa(i), user.UserId)
		assert.Equal(t, []string{strconv.Itoa(i + 100)}, user.Labels)
		assert.Equal(t, fmt.Sprintf("comment %d", i), user.Comment)
	}
	// Get this user
	if user, err := db.GetUser("0"); err != nil {
		t.Fatal(err)
	} else {
		assert.Equal(t, "0", user.UserId)
	}
	// Delete this user
	err := db.DeleteUser("0")
	assert.Nil(t, err)
	_, err = db.GetUser("0")
	assert.NotNil(t, err)
	// test override
	err = db.InsertUser(User{UserId: "1", Comment: "override"})
	assert.Nil(t, err)
	user, err := db.GetUser("1")
	assert.Nil(t, err)
	assert.Equal(t, "override", user.Comment)
}

func testFeedback(t *testing.T, db Database) {
	// users that already exists
	err := db.InsertUser(User{"0", []string{"a"}, []string{"x"}, "comment"})
	assert.Nil(t, err)
	// items that already exists
	err = db.InsertItem(Item{ItemId: "0", Labels: []string{"b"}})
	assert.Nil(t, err)
	// Insert ret
	feedback := []Feedback{
		{FeedbackKey{positiveFeedbackType, "0", "0"}, time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC), "comment"},
		{FeedbackKey{positiveFeedbackType, "1", "2"}, time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC), "comment"},
		{FeedbackKey{positiveFeedbackType, "2", "4"}, time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC), "comment"},
		{FeedbackKey{positiveFeedbackType, "3", "6"}, time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC), "comment"},
		{FeedbackKey{positiveFeedbackType, "4", "8"}, time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC), "comment"},
	}
	err = db.BatchInsertFeedback(feedback[1:], true, true)
	assert.Nil(t, err)
	err = db.InsertFeedback(feedback[0], true, true)
	assert.Nil(t, err)
	// other type
	err = db.InsertFeedback(Feedback{FeedbackKey: FeedbackKey{negativeFeedbackType, "0", "2"}}, true, true)
	assert.Nil(t, err)
	err = db.InsertFeedback(Feedback{FeedbackKey: FeedbackKey{negativeFeedbackType, "2", "4"}}, true, true)
	assert.Nil(t, err)
	// Get feedback
	ret := getFeedback(t, db, &positiveFeedbackType)
	assert.Equal(t, feedback, ret)
	ret = getFeedback(t, db, nil)
	assert.Equal(t, len(feedback)+2, len(ret))
	// Get items
	items := getItems(t, db)
	assert.Equal(t, 5, len(items))
	for i, item := range items {
		assert.Equal(t, strconv.Itoa(i*2), item.ItemId)
	}
	// Get users
	users := getUsers(t, db)
	assert.Equal(t, 5, len(users))
	for i, user := range users {
		assert.Equal(t, strconv.Itoa(i), user.UserId)
	}
	// check users that already exists
	user, err := db.GetUser("0")
	assert.Nil(t, err)
	assert.Equal(t, User{"0", []string{"a"}, []string{"x"}, "comment"}, user)
	// check items that already exists
	item, err := db.GetItem("0")
	assert.Nil(t, err)
	assert.Equal(t, Item{ItemId: "0", Labels: []string{"b"}}, item)
	// Get typed feedback by user
	ret, err = db.GetUserFeedback("2", &positiveFeedbackType)
	assert.Nil(t, err)
	assert.Equal(t, 1, len(ret))
	assert.Equal(t, "2", ret[0].UserId)
	assert.Equal(t, "4", ret[0].ItemId)
	// Get all feedback by user
	ret, err = db.GetUserFeedback("2", nil)
	assert.Nil(t, err)
	assert.Equal(t, 2, len(ret))
	// Get typed feedback by item
	ret, err = db.GetItemFeedback("4", &positiveFeedbackType)
	assert.Nil(t, err)
	assert.Equal(t, 1, len(ret))
	assert.Equal(t, "2", ret[0].UserId)
	assert.Equal(t, "4", ret[0].ItemId)
	// Get all feedback by item
	ret, err = db.GetItemFeedback("4", nil)
	assert.Nil(t, err)
	assert.Equal(t, 2, len(ret))
	// test override
	err = db.InsertFeedback(Feedback{
		FeedbackKey: FeedbackKey{positiveFeedbackType, "0", "0"},
		Comment:     "override",
	}, true, true)
	assert.Nil(t, err)
	ret, err = db.GetUserFeedback("0", &positiveFeedbackType)
	assert.Nil(t, err)
	assert.Equal(t, 1, len(ret))
	assert.Equal(t, "override", ret[0].Comment)
}

func testItems(t *testing.T, db Database) {
	// Items
	items := []Item{
		{
			ItemId:    "0",
			Timestamp: time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC),
			Labels:    []string{"a"},
			Comment:   "comment 0",
		},
		{
			ItemId:    "2",
			Timestamp: time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC),
			Labels:    []string{"a"},
			Comment:   "comment 2",
		},
		{
			ItemId:    "4",
			Timestamp: time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC),
			Labels:    []string{"a", "b"},
			Comment:   "comment 4",
		},
		{
			ItemId:    "6",
			Timestamp: time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC),
			Labels:    []string{"b"},
			Comment:   "comment 6",
		},
		{
			ItemId:    "8",
			Timestamp: time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC),
			Labels:    []string{"b"},
			Comment:   "comment 8",
		},
	}
	// Insert item
	err := db.BatchInsertItem(items[1:])
	assert.Nil(t, err)
	err = db.InsertItem(items[0])
	assert.Nil(t, err)
	// Get items
	totalItems := getItems(t, db)
	assert.Equal(t, items, totalItems)
	// Get item
	for _, item := range items {
		ret, err := db.GetItem(item.ItemId)
		if err != nil {
			t.Fatal(err)
		}
		assert.Equal(t, item, ret)
	}
	// Delete item
	err = db.DeleteItem("0")
	assert.Nil(t, err)
	_, err = db.GetItem("0")
	assert.NotNil(t, err)
	// test override
	err = db.InsertItem(Item{ItemId: "2", Comment: "override"})
	assert.Nil(t, err)
	item, err := db.GetItem("2")
	assert.Nil(t, err)
	assert.Equal(t, "override", item.Comment)
}

func testDeleteUser(t *testing.T, db Database) {
	// Insert ret
	feedback := []Feedback{
		{FeedbackKey{positiveFeedbackType, "0", "0"}, time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC), "comment"},
		{FeedbackKey{positiveFeedbackType, "0", "2"}, time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC), "comment"},
		{FeedbackKey{positiveFeedbackType, "0", "4"}, time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC), "comment"},
		{FeedbackKey{positiveFeedbackType, "0", "6"}, time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC), "comment"},
		{FeedbackKey{positiveFeedbackType, "0", "8"}, time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC), "comment"},
	}
	err := db.BatchInsertFeedback(feedback, true, true)
	assert.Nil(t, err)
	// Delete user
	err = db.DeleteUser("0")
	assert.Nil(t, err)
	_, err = db.GetUser("0")
	assert.NotNil(t, err, "failed to delete user")
	ret, err := db.GetUserFeedback("0", &positiveFeedbackType)
	assert.Nil(t, err)
	assert.Equal(t, 0, len(ret))
	_, ret, err = db.GetFeedback("", 100, &positiveFeedbackType)
	assert.Nil(t, err)
	assert.Empty(t, ret)
}

func testDeleteItem(t *testing.T, db Database) {
	// Insert ret
	feedbacks := []Feedback{
		{FeedbackKey{positiveFeedbackType, "0", "0"}, time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC), "comment"},
		{FeedbackKey{positiveFeedbackType, "1", "0"}, time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC), "comment"},
		{FeedbackKey{positiveFeedbackType, "2", "0"}, time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC), "comment"},
		{FeedbackKey{positiveFeedbackType, "3", "0"}, time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC), "comment"},
		{FeedbackKey{positiveFeedbackType, "4", "0"}, time.Date(1996, 3, 15, 0, 0, 0, 0, time.UTC), "comment"},
	}
	err := db.BatchInsertFeedback(feedbacks, true, true)
	assert.Nil(t, err)
	// Delete item
	err = db.DeleteItem("0")
	assert.Nil(t, err)
	_, err = db.GetItem("0")
	assert.NotNil(t, err, "failed to delete item")
	ret, err := db.GetItemFeedback("0", &positiveFeedbackType)
	assert.Nil(t, err)
	assert.Equal(t, 0, len(ret))
	_, ret, err = db.GetFeedback("", 100, &positiveFeedbackType)
	assert.Nil(t, err)
	assert.Empty(t, ret)
}
