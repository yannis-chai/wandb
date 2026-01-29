package leet

import (
	"errors"
	"io"
	"slices"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	spb "github.com/wandb/wandb/core/pkg/service_go_proto"
)

const (
	// Boot loading parameters
	bootLoadChunkSize = 1000
	bootLoadMaxTime   = 100 * time.Millisecond

	// Live monitoring parameters
	liveMonitorChunkSize = 2000
	liveMonitorMaxTime   = 50 * time.Millisecond
)

// HistorySource is an interface for reading W&B run history data.
//
// Implementations:
//   - LevelDBHistorySource: Reads from a LevelDB-style .wandb transaction log
//   - ParquetHistorySource: Reads from a run's exported parquet history files.
//     - The files are downloaded from the W&B backend.
//
// The Read method returns a ChunkedBatchMsg containing processed records,
// and may return io.EOF when the stream is complete.
type HistorySource interface {
	// Read reads events from the history source,
	// up to a given number of records or a given time period,
	// whichever is reached first.
	//
	// Returns a ChunkedBatchMsg with processed records and metadata.
	// If the history source has been completely read, it returns io.EOF error.
	Read(
		chunkSize int,
		maxTimePerChunk time.Duration,
	) (tea.Msg, error)

	// Close closes the history source that is being read from.
	Close()
}

// ReadAllRecordsChunked returns a command to read records in chunks for progressive loading.
func ReadAllRecordsChunked(source HistorySource) tea.Cmd {
	return func() tea.Msg {
		msgs, err := source.Read(
			bootLoadChunkSize,
			bootLoadMaxTime,
		)
		if err != nil && !errors.Is(err, io.EOF) {
			return ErrorMsg{Err: err}
		}
		return msgs
	}
}

// ReadAvailableRecords reads new records for live monitoring.
func ReadAvailableRecords(source HistorySource) tea.Cmd {
	return func() tea.Msg {
		msgs, err := source.Read(
			liveMonitorChunkSize,
			liveMonitorMaxTime,
		)

		// For live monitoring, we ignore EOF errors
		// since we may have more data to read later.
		if err != nil && !errors.Is(err, io.EOF) {
			return ErrorMsg{Err: err}
		}
		return msgs
	}
}

// concatenateHistory merges a slice of HistoryMsg into a single HistoryMsg.
//
// Assumes that the history messages are ordered.
func concatenateHistory(messages []HistoryMsg) HistoryMsg {
	h := HistoryMsg{
		Metrics: make(map[string]MetricData),
	}

	for _, msg := range messages {
		for metricName, data := range msg.Metrics {
			existing := h.Metrics[metricName]
			h.Metrics[metricName] = MetricData{
				X: slices.Concat(existing.X, data.X),
				Y: slices.Concat(existing.Y, data.Y),
			}
		}
	}

	return h
}

// concatenateSummary merges a slice of SummaryMsg into a single SummaryMsg.
//
// Assumes that the summary messages are ordered.
func concatenateSummary(messages []SummaryMsg) SummaryMsg {
	s := SummaryMsg{
		Summary: make([]*spb.SummaryRecord, 0),
	}

	for _, msg := range messages {
		s.Summary = append(s.Summary, msg.Summary...)
	}

	return s
}
