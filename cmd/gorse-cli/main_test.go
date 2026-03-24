// Copyright 2026 gorse Project Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/emicklei/go-restful/v3"
	"github.com/gorse-io/gorse/config"
	"github.com/gorse-io/gorse/master"
	"github.com/gorse-io/gorse/storage/cache"
	"github.com/gorse-io/gorse/storage/data"
	"github.com/gorse-io/gorse/storage/meta"
	"github.com/stretchr/testify/suite"
)

const (
	testAPIKey = "test-api-key"
)

type CLITestSuite struct {
	suite.Suite
	master        master.Master
	handler       *restful.Container
	server        *httptest.Server
	tempDir       string
	originalArgs  []string
}

func (suite *CLITestSuite) SetupTest() {
	var err error
	suite.tempDir = suite.T().TempDir()

	// Setup config
	suite.master.Config = config.GetDefaultConfig()
	suite.master.Config.Master.APIKey = testAPIKey
	suite.master.Config.Database.DataStore = fmt.Sprintf("sqlite://%s/data.db", suite.tempDir)
	suite.master.Config.Database.CacheStore = fmt.Sprintf("sqlite://%s/cache.db", suite.tempDir)

	// Open database
	suite.master.metaStore, err = meta.Open(fmt.Sprintf("sqlite://%s/meta.db", suite.tempDir), suite.master.Config.Master.MetaTimeout)
	suite.NoError(err)
	suite.master.DataClient, err = data.Open(suite.master.Config.Database.DataStore, "")
	suite.NoError(err)
	suite.master.CacheClient, err = cache.Open(suite.master.Config.Database.CacheStore, "")
	suite.NoError(err)

	// Init database
	err = suite.master.metaStore.Init()
	suite.NoError(err)
	err = suite.master.DataClient.Init()
	suite.NoError(err)
	err = suite.master.CacheClient.Init()
	suite.NoError(err)

	// Create web service
	suite.master.WebService = new(restful.WebService)
	suite.master.CreateWebService()
	suite.master.RestServer.CreateWebService()
	suite.master.cancel = func() {}
	suite.master.scheduled = make(chan struct{}, 1)

	// Create handler
	suite.handler = restful.NewContainer()
	suite.handler.Add(suite.master.WebService)

	// Create test server
	suite.server = httptest.NewServer(suite.handler)

	// Save original args
	suite.originalArgs = os.Args
}

func (suite *CLITestSuite) TearDownTest() {
	suite.server.Close()
	err := suite.master.metaStore.Close()
	suite.NoError(err)
	err = suite.master.DataClient.Close()
	suite.NoError(err)
	err = suite.master.CacheClient.Close()
	suite.NoError(err)
	os.Args = suite.originalArgs
}

func (suite *CLITestSuite) TestGetTasks() {
	// Set environment variables
	os.Setenv("GORSE_ADMIN_ENDPOINT", suite.server.URL)
	os.Setenv("GORSE_ADMIN_API_KEY", testAPIKey)
	defer os.Unsetenv("GORSE_ADMIN_ENDPOINT")
	defer os.Unsetenv("GORSE_ADMIN_API_KEY")

	// Capture output
	var buf bytes.Buffer
	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	// Run command
	os.Args = []string{"gorse-cli", "get", "tasks"}
	err := rootCmd.Execute()
	suite.NoError(err)

	// Restore stdout
	w.Close()
	os.Stdout = oldStdout
	buf.ReadFrom(r)

	// Verify output is valid JSON
	var result interface{}
	err = json.Unmarshal(buf.Bytes(), &result)
	suite.NoError(err, "Output should be valid JSON: %s", buf.String())
}

func (suite *CLITestSuite) TestGetConfig() {
	// Set environment variables
	os.Setenv("GORSE_ADMIN_ENDPOINT", suite.server.URL)
	os.Setenv("GORSE_ADMIN_API_KEY", testAPIKey)
	defer os.Unsetenv("GORSE_ADMIN_ENDPOINT")
	defer os.Unsetenv("GORSE_ADMIN_API_KEY")

	// Capture output
	var buf bytes.Buffer
	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	// Run command
	os.Args = []string{"gorse-cli", "get", "config"}
	err := rootCmd.Execute()
	suite.NoError(err)

	// Restore stdout
	w.Close()
	os.Stdout = oldStdout
	buf.ReadFrom(r)

	// Verify output is valid JSON
	var result map[string]interface{}
	err = json.Unmarshal(buf.Bytes(), &result)
	suite.NoError(err, "Output should be valid JSON: %s", buf.String())
}

func (suite *CLITestSuite) TestSetConfig() {
	// Set environment variables
	os.Setenv("GORSE_ADMIN_ENDPOINT", suite.server.URL)
	os.Setenv("GORSE_ADMIN_API_KEY", testAPIKey)
	defer os.Unsetenv("GORSE_ADMIN_ENDPOINT")
	defer os.Unsetenv("GORSE_ADMIN_API_KEY")

	// Capture output
	var buf bytes.Buffer
	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	// Run command
	os.Args = []string{"gorse-cli", "set", "config", "recommend.cache_size=1000"}
	err := rootCmd.Execute()
	suite.NoError(err)

	// Restore stdout
	w.Close()
	os.Stdout = oldStdout
	buf.ReadFrom(r)

	// Verify output is valid JSON
	var result map[string]interface{}
	err = json.Unmarshal(buf.Bytes(), &result)
	suite.NoError(err, "Output should be valid JSON: %s", buf.String())
}

func (suite *CLITestSuite) TestGetTasksWithFlags() {
	// Capture output
	var buf bytes.Buffer
	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	// Run command with flags
	os.Args = []string{"gorse-cli", "get", "tasks",
		"--endpoint", suite.server.URL,
		"--api-key", testAPIKey}
	err := rootCmd.Execute()
	suite.NoError(err)

	// Restore stdout
	w.Close()
	os.Stdout = oldStdout
	buf.ReadFrom(r)

	// Verify output is valid JSON
	var result interface{}
	err = json.Unmarshal(buf.Bytes(), &result)
	suite.NoError(err, "Output should be valid JSON: %s", buf.String())
}

func (suite *CLITestSuite) TestMissingEndpoint() {
	// Clear environment variables
	os.Unsetenv("GORSE_ADMIN_ENDPOINT")
	os.Unsetenv("GORSE_ADMIN_API_KEY")

	// Run command without endpoint
	os.Args = []string{"gorse-cli", "get", "tasks"}
	err := rootCmd.Execute()
	// Should fail because endpoint is missing
	suite.Error(err)
	suite.True(strings.Contains(err.Error(), "endpoint") || strings.Contains(err.Error(), "required"))
}

func (suite *CLITestSuite) TestInvalidAPIKey() {
	// Set environment variables with wrong API key
	os.Setenv("GORSE_ADMIN_ENDPOINT", suite.server.URL)
	os.Setenv("GORSE_ADMIN_API_KEY", "wrong-key")
	defer os.Unsetenv("GORSE_ADMIN_ENDPOINT")
	defer os.Unsetenv("GORSE_ADMIN_API_KEY")

	// Run command
	os.Args = []string{"gorse-cli", "get", "tasks"}
	err := rootCmd.Execute()
	// Should fail because API key is wrong
	suite.Error(err)
}

func TestCLISuite(t *testing.T) {
	suite.Run(t, new(CLITestSuite))
}
