/**
 * MistralCoreTests.swift
 * Unit tests for MistralCore
 */

import XCTest
@testable import MistralCore

final class MistralCoreTests: XCTestCase {
    func testTokenizer() throws {
        let tokenizer = TekkenTokenizer()
        let text = "Hello world"
        let tokens = tokenizer.encode(text)
        XCTAssertFalse(tokens.isEmpty, "Tokenization should produce tokens")
    }

    func testHiddenStatesConfig() throws {
        let config = HiddenStatesConfig.mfluxDefault
        XCTAssertEqual(config.layerIndices, [10, 20, 30])
        XCTAssertTrue(config.concatenate)
    }

    func testModelRegistry() throws {
        let models = ModelRegistry.shared.allModels()
        XCTAssertEqual(models.count, 3, "Should have 3 model variants")

        let defaultModel = ModelRegistry.shared.defaultModel()
        XCTAssertEqual(defaultModel.variant, .mlx8bit)
    }

    func testGenerateParameters() throws {
        let params = GenerateParameters.balanced
        XCTAssertEqual(params.maxTokens, 512)
        XCTAssertEqual(params.temperature, 0.7)
        XCTAssertEqual(params.topP, 0.9)
    }
}
