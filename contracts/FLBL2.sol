// SPDX-License-Identifier: MIT
pragma solidity 0.8.20;

// OpenZeppelin Contracts v5.0.2
import "https://github.com/OpenZeppelin/openzeppelin-contracts/blob/v5.0.2/contracts/access/Ownable.sol";
import "https://github.com/OpenZeppelin/openzeppelin-contracts/blob/v5.0.2/contracts/utils/Pausable.sol";

/**
 * @title   FLBL2s
 * @notice  Immutable audit ledger for federated learning rounds on Base mainnet (EVM L2).
 */
contract FLBL2 is Ownable, Pausable {

    error NotAuthorized(address caller);
    error ClientsCanOnlyAddLocalBlocks(string attempted);
    error BlockDoesNotExist(uint256 index, uint256 chainLength);

    struct Block {
        uint256 blockNumber;
        uint256 flRound;
        string  blockType;
        bytes32 contentHash;
        bytes32 previousHash;
        uint256 timestamp;
        address submitter;
    }

    Block[] public blocks;
    mapping(address => bool) public authorizedClients;

    event BlockAdded(uint256 indexed blockNumber, uint256 indexed flRound, string blockType, bytes32 contentHash);
    event ClientAuthorized(address indexed client);
    event ClientRevoked(address indexed client);

    /**
     * @dev In OZ v5, the Ownable constructor REQUIRES an initial owner address.
     * Use msg.sender to set the deployer as the initial owner.
     */
    constructor() Ownable(msg.sender) {
        blocks.push(Block({
            blockNumber:  0,
            flRound:      0,
            blockType:    "GENESIS",
            contentHash:  keccak256("FL Blockchain Genesis"),
            previousHash: bytes32(0),
            timestamp:    block.timestamp,
            submitter:    msg.sender
        }));
    }

    function authorizeClient(address client) external onlyOwner {
        authorizedClients[client] = true;
        emit ClientAuthorized(client);
    }

    function revokeClient(address client) external onlyOwner {
        authorizedClients[client] = false;
        emit ClientRevoked(client);
    }

    function pause()   external onlyOwner { _pause(); }
    function unpause() external onlyOwner { _unpause(); }

    function addBlock(uint256 flRound, string memory blockType, bytes memory data) external whenNotPaused returns (uint256) {
        if (msg.sender != owner()) {
            if (!authorizedClients[msg.sender]) revert NotAuthorized(msg.sender);
            if (keccak256(bytes(blockType)) != keccak256(bytes("LOCAL"))) revert ClientsCanOnlyAddLocalBlocks(blockType);
        }

        bytes32 contentHash  = keccak256(data);
        bytes32 previousHash = blocks[blocks.length - 1].contentHash;

        blocks.push(Block({
            blockNumber:  blocks.length,
            flRound:      flRound,
            blockType:    blockType,
            contentHash:  contentHash,
            previousHash: previousHash,
            timestamp:    block.timestamp,
            submitter:    msg.sender
        }));

        emit BlockAdded(blocks.length - 1, flRound, blockType, contentHash);
        return blocks.length - 1;
    }

    function getBlockCount() external view returns (uint256) { return blocks.length; }
    
    function verifyChain() external view returns (bool) {
        for (uint256 i = 1; i < blocks.length; i++) {
            if (blocks[i].previousHash != blocks[i - 1].contentHash) return false;
        }
        return true;
    }
}