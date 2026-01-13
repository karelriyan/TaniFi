// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {ERC721} from "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";

/// @title FarmerRegistry - Soulbound Identity Token for TaniFi Farmers
/// @notice Non-transferable NFT representing farmer identity and reputation
/// @dev Implements Soulbound Token (SBT) pattern - tokens cannot be transferred after minting
/// @author TaniFi Team - Lisk Builders Challenge 2026

contract FarmerRegistry is ERC721, Ownable {
    // ============ Structs ============

    struct FarmerProfile {
        uint256 tokenId;
        address walletAddress;
        bytes32 phoneHash;          // Hash of phone number for privacy
        uint256 reputationScore;    // On-chain reputation (starts at 100)
        bool isKYCVerified;
        uint256 registeredAt;
        uint256 completedProjects;
        uint256 failedProjects;
        string metadataURI;         // IPFS hash with additional data
    }

    // ============ State Variables ============

    address public taniVault;       // TaniVault contract address for reputation updates

    mapping(address => bool) public kycAdmins;

    // Token storage
    uint256 private _tokenIdCounter;
    mapping(address => uint256) public tokenOfOwner;  // One token per farmer
    mapping(uint256 => FarmerProfile) public profiles;

    // Phone hash to wallet mapping (prevent duplicate registrations)
    mapping(bytes32 => address) public phoneHashToWallet;

    // Reputation constants
    uint256 public constant INITIAL_REPUTATION = 100;
    uint256 public constant MAX_REPUTATION = 1000;
    uint256 public constant PROJECT_SUCCESS_BONUS = 10;
    uint256 public constant PROJECT_FAILURE_PENALTY = 20;

    // ============ Events ============

    event FarmerRegistered(
        uint256 indexed tokenId,
        address indexed farmer,
        bytes32 phoneHash
    );
    event FarmerVerified(address indexed farmer, uint256 tokenId);
    event ReputationUpdated(
        address indexed farmer,
        uint256 oldScore,
        uint256 newScore,
        string reason
    );
    event MetadataUpdated(uint256 indexed tokenId, string newURI);
    event KYCAdminAdded(address indexed admin);
    event KYCAdminRemoved(address indexed admin);
    event TaniVaultUpdated(address indexed newTaniVault);

    // ============ Errors ============

    error NotKYCAdmin();
    error NotTaniVault();
    error InvalidAddress();
    error AlreadyRegistered();
    error PhoneAlreadyUsed();
    error FarmerNotRegistered();
    error AlreadyVerified();
    error ScoreTooHigh();
    error SoulboundTransferBlocked();

    // ============ Modifiers ============

    modifier onlyKYCAdmin() {
        if (!kycAdmins[msg.sender] && msg.sender != owner()) revert NotKYCAdmin();
        _;
    }

    modifier onlyTaniVault() {
        if (msg.sender != taniVault) revert NotTaniVault();
        _;
    }

    // ============ Constructor ============

    constructor() ERC721("TaniFi Farmer Identity", "TANI-ID") Ownable(msg.sender) {
        kycAdmins[msg.sender] = true;
    }

    // ============ Soulbound Override ============

    /// @notice Override to prevent transfers - Soulbound tokens
    /// @dev This function is called before any token transfer
    function _update(address to, uint256 tokenId, address auth) internal override returns (address) {
        address from = _ownerOf(tokenId);

        // Allow minting (from == address(0)) but block all transfers
        if (from != address(0) && to != address(0)) {
            revert SoulboundTransferBlocked();
        }

        return super._update(to, tokenId, auth);
    }

    // ============ Registration Functions ============

    /// @notice Register a new farmer (mints SBT)
    /// @param _farmer Wallet address of the farmer
    /// @param _phoneHash Hash of farmer's phone number
    /// @param _metadataURI IPFS URI with additional farmer data
    function registerFarmer(
        address _farmer,
        bytes32 _phoneHash,
        string calldata _metadataURI
    ) external onlyKYCAdmin returns (uint256) {
        if (_farmer == address(0)) revert InvalidAddress();
        if (tokenOfOwner[_farmer] != 0) revert AlreadyRegistered();
        if (phoneHashToWallet[_phoneHash] != address(0)) revert PhoneAlreadyUsed();

        _tokenIdCounter++;
        uint256 tokenId = _tokenIdCounter;

        _safeMint(_farmer, tokenId);
        tokenOfOwner[_farmer] = tokenId;
        phoneHashToWallet[_phoneHash] = _farmer;

        profiles[tokenId] = FarmerProfile({
            tokenId: tokenId,
            walletAddress: _farmer,
            phoneHash: _phoneHash,
            reputationScore: INITIAL_REPUTATION,
            isKYCVerified: false,
            registeredAt: block.timestamp,
            completedProjects: 0,
            failedProjects: 0,
            metadataURI: _metadataURI
        });

        emit FarmerRegistered(tokenId, _farmer, _phoneHash);

        return tokenId;
    }

    /// @notice Verify farmer's KYC status
    /// @param _farmer Address of the farmer to verify
    function verifyFarmer(address _farmer) external onlyKYCAdmin {
        uint256 tokenId = tokenOfOwner[_farmer];
        if (tokenId == 0) revert FarmerNotRegistered();
        if (profiles[tokenId].isKYCVerified) revert AlreadyVerified();

        profiles[tokenId].isKYCVerified = true;

        emit FarmerVerified(_farmer, tokenId);
    }

    /// @notice Revoke farmer's KYC verification
    /// @param _farmer Address of the farmer
    function revokeVerification(address _farmer) external onlyKYCAdmin {
        uint256 tokenId = tokenOfOwner[_farmer];
        if (tokenId == 0) revert FarmerNotRegistered();

        profiles[tokenId].isKYCVerified = false;
    }

    // ============ Reputation Functions ============

    /// @notice Update reputation after project completion (called by TaniVault)
    /// @param _farmer Address of the farmer
    /// @param _success Whether the project was successful
    function recordProjectCompletion(address _farmer, bool _success) external onlyTaniVault {
        uint256 tokenId = tokenOfOwner[_farmer];
        if (tokenId == 0) revert FarmerNotRegistered();

        FarmerProfile storage profile = profiles[tokenId];
        uint256 oldScore = profile.reputationScore;

        if (_success) {
            profile.completedProjects++;
            // Increase reputation (capped at MAX)
            uint256 newScore = oldScore + PROJECT_SUCCESS_BONUS;
            profile.reputationScore = newScore > MAX_REPUTATION ? MAX_REPUTATION : newScore;
            emit ReputationUpdated(_farmer, oldScore, profile.reputationScore, "PROJECT_SUCCESS");
        } else {
            profile.failedProjects++;
            // Decrease reputation (floor at 0)
            if (oldScore > PROJECT_FAILURE_PENALTY) {
                profile.reputationScore = oldScore - PROJECT_FAILURE_PENALTY;
            } else {
                profile.reputationScore = 0;
            }
            emit ReputationUpdated(_farmer, oldScore, profile.reputationScore, "PROJECT_FAILURE");
        }
    }

    /// @notice Manual reputation adjustment (admin only, for special cases)
    /// @param _farmer Address of the farmer
    /// @param _newScore New reputation score
    /// @param _reason Reason for adjustment
    function adjustReputation(
        address _farmer,
        uint256 _newScore,
        string calldata _reason
    ) external onlyOwner {
        uint256 tokenId = tokenOfOwner[_farmer];
        if (tokenId == 0) revert FarmerNotRegistered();
        if (_newScore > MAX_REPUTATION) revert ScoreTooHigh();

        uint256 oldScore = profiles[tokenId].reputationScore;
        profiles[tokenId].reputationScore = _newScore;

        emit ReputationUpdated(_farmer, oldScore, _newScore, _reason);
    }

    // ============ Metadata Functions ============

    /// @notice Update farmer's metadata URI
    /// @param _farmer Address of the farmer
    /// @param _newURI New IPFS URI
    function updateMetadata(address _farmer, string calldata _newURI) external onlyKYCAdmin {
        uint256 tokenId = tokenOfOwner[_farmer];
        if (tokenId == 0) revert FarmerNotRegistered();

        profiles[tokenId].metadataURI = _newURI;

        emit MetadataUpdated(tokenId, _newURI);
    }

    /// @notice Get token metadata URI (ERC721 compatible)
    /// @param tokenId Token ID
    function tokenURI(uint256 tokenId) public view override returns (string memory) {
        _requireOwned(tokenId);
        return profiles[tokenId].metadataURI;
    }

    // ============ View Functions ============

    /// @notice Get total supply
    function totalSupply() external view returns (uint256) {
        return _tokenIdCounter;
    }

    /// @notice Get full farmer profile
    /// @param _farmer Address of the farmer
    function getFarmerProfile(address _farmer) external view returns (FarmerProfile memory) {
        uint256 tokenId = tokenOfOwner[_farmer];
        if (tokenId == 0) revert FarmerNotRegistered();
        return profiles[tokenId];
    }

    /// @notice Get farmer's reputation score
    /// @param _farmer Address of the farmer
    function getReputation(address _farmer) external view returns (uint256) {
        uint256 tokenId = tokenOfOwner[_farmer];
        if (tokenId == 0) return 0;
        return profiles[tokenId].reputationScore;
    }

    /// @notice Check if farmer is registered
    /// @param _farmer Address to check
    function isRegistered(address _farmer) external view returns (bool) {
        return tokenOfOwner[_farmer] != 0;
    }

    /// @notice Check if farmer is KYC verified
    /// @param _farmer Address to check
    function isVerified(address _farmer) external view returns (bool) {
        uint256 tokenId = tokenOfOwner[_farmer];
        if (tokenId == 0) return false;
        return profiles[tokenId].isKYCVerified;
    }

    // ============ Admin Functions ============

    function setTaniVault(address _taniVault) external onlyOwner {
        if (_taniVault == address(0)) revert InvalidAddress();
        taniVault = _taniVault;
        emit TaniVaultUpdated(_taniVault);
    }

    function addKYCAdmin(address _admin) external onlyOwner {
        if (_admin == address(0)) revert InvalidAddress();
        kycAdmins[_admin] = true;
        emit KYCAdminAdded(_admin);
    }

    function removeKYCAdmin(address _admin) external onlyOwner {
        kycAdmins[_admin] = false;
        emit KYCAdminRemoved(_admin);
    }
}
