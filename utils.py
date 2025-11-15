"""
Utility functions for message management and token optimization.
"""
from typing import List, Set
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolMessage, HumanMessage


def truncate_messages(
    messages: List[BaseMessage],
    system_message: SystemMessage | None = None,
    max_messages: int = 20,
    max_tokens_approx: int = 100000,  # Conservative default to stay under 200k TPM limit
) -> List[BaseMessage]:
    """
    Truncate messages to prevent token limit issues while preserving tool call sequences.
    
    This function keeps the system message (if provided) and the most recent
    messages, ensuring we don't exceed token limits for rate limiting.
    Critically, it preserves the integrity of tool call sequences: if an AIMessage
    has tool_calls, all corresponding ToolMessages are kept together.
    
    Args:
        messages: List of messages to truncate
        system_message: Optional system message to always keep at the beginning
        max_messages: Maximum number of messages to keep (excluding system message)
        max_tokens_approx: Approximate maximum tokens (rough estimate: ~4 chars per token)
    
    Returns:
        Truncated list of messages with preserved tool call sequences
    """
    # Rough token estimation: approximately 4 characters per token
    chars_per_token = 4
    
    def get_message_tokens(msg: BaseMessage) -> int:
        """Estimate token count for a message."""
        content = ""
        if hasattr(msg, 'content'):
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                content = str(msg.content)
        return len(content) // chars_per_token if content else 0
    
    # Start with system message if provided
    result = []
    if system_message:
        result.append(system_message)
    
    # Calculate approximate tokens for system message
    system_tokens = get_message_tokens(system_message) if system_message else 0
    
    # Filter out system messages from the input (we handle them separately)
    filtered_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
    
    if not filtered_messages:
        return result
    
    # Group messages into sequences: each AIMessage with tool_calls + its ToolMessages
    # This ensures we never break tool call sequences
    sequences: List[tuple[List[BaseMessage], bool]] = []  # (sequence, is_complete)
    current_sequence: List[BaseMessage] = []
    pending_tool_call_ids: Set[str] = set()
    
    for message in filtered_messages:
        if isinstance(message, AIMessage):
            # If we have a pending sequence, check if it's complete
            if current_sequence:
                # Sequence is complete only if all tool_call_ids were matched
                is_complete = len(pending_tool_call_ids) == 0
                sequences.append((current_sequence, is_complete))
                current_sequence = []
                pending_tool_call_ids = set()
            
            # Check if this AI message has tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                current_sequence.append(message)
                # Extract all tool_call_ids
                for tool_call in message.tool_calls:
                    if hasattr(tool_call, 'id') and tool_call.id:
                        pending_tool_call_ids.add(tool_call.id)
            else:
                # Regular AI message without tool calls - complete sequence
                sequences.append(([message], True))
        
        elif isinstance(message, ToolMessage):
            # Check if this tool message belongs to the current sequence
            if (hasattr(message, 'tool_call_id') and 
                message.tool_call_id and 
                message.tool_call_id in pending_tool_call_ids):
                current_sequence.append(message)
                pending_tool_call_ids.discard(message.tool_call_id)
            else:
                # This tool message doesn't belong to current sequence
                # Save current sequence (may be incomplete)
                if current_sequence:
                    is_complete = len(pending_tool_call_ids) == 0
                    sequences.append((current_sequence, is_complete))
                # Orphaned tool messages form their own complete sequence
                sequences.append(([message], True))
                current_sequence = []
                pending_tool_call_ids = set()
        else:
            # Other message types (HumanMessage, etc.)
            # If we have a pending sequence, save it first
            if current_sequence:
                is_complete = len(pending_tool_call_ids) == 0
                sequences.append((current_sequence, is_complete))
                current_sequence = []
                pending_tool_call_ids = set()
            # Other messages are complete sequences
            sequences.append(([message], True))
    
    # Don't forget the last sequence
    if current_sequence:
        is_complete = len(pending_tool_call_ids) == 0
        sequences.append((current_sequence, is_complete))
    
    # Now work backwards from the end, including only COMPLETE sequences
    messages_to_add: List[BaseMessage] = []
    total_tokens_approx = system_tokens
    message_count = 0
    
    # Start from the last sequence and work backwards
    for seq, is_complete in reversed(sequences):
        # Only include sequences that are complete
        # Incomplete sequences (AIMessage with tool_calls but missing ToolMessages) will cause API errors
        if not is_complete:
            # Check if this is a sequence with an AIMessage that has tool_calls
            # If so, we must skip it to avoid API errors
            has_ai_with_tool_calls = any(
                isinstance(msg, AIMessage) and 
                hasattr(msg, 'tool_calls') and 
                msg.tool_calls 
                for msg in seq
            )
            if has_ai_with_tool_calls:
                # Skip incomplete tool call sequences
                continue
        
        # Calculate tokens for this entire sequence
        seq_tokens = sum(get_message_tokens(msg) for msg in seq)
        seq_message_count = len(seq)
        
        # Skip sequences with extremely large individual messages
        if any(get_message_tokens(msg) > 30000 for msg in seq):
            if messages_to_add:  # Only skip if we have other messages
                continue
        
        # Check if we can fit this sequence
        if (message_count + seq_message_count <= max_messages and
            total_tokens_approx + seq_tokens <= max_tokens_approx):
            # Add the sequence (it will be reversed later)
            messages_to_add.extend(seq)
            total_tokens_approx += seq_tokens
            message_count += seq_message_count
        else:
            # Can't fit this sequence, stop here
            break
    
    # Reverse to get chronological order (oldest to newest)
    messages_to_add.reverse()
    
    # Build a comprehensive map of all ToolMessages by their tool_call_id
    # This helps us quickly check if all required tool_call_ids have responses
    all_tool_messages_by_id: dict[str, ToolMessage] = {}
    for msg in messages_to_add:
        if isinstance(msg, ToolMessage):
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                all_tool_messages_by_id[msg.tool_call_id] = msg
    
    # Final validation: Only include AIMessages with tool_calls if ALL tool_call_ids have ToolMessages
    validated_messages: List[BaseMessage] = []
    i = 0
    while i < len(messages_to_add):
        msg = messages_to_add[i]
        
        # Check if this is an AIMessage with tool_calls
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            # Extract all tool_call_ids from this AIMessage
            required_tool_call_ids = set()
            for tool_call in msg.tool_calls:
                # Handle both dict-like and object-like tool_call structures
                tool_call_id = None
                if hasattr(tool_call, 'id'):
                    tool_call_id = tool_call.id
                elif isinstance(tool_call, dict):
                    tool_call_id = tool_call.get('id')
                
                if tool_call_id:
                    required_tool_call_ids.add(tool_call_id)
            
            if not required_tool_call_ids:
                # No tool_call_ids found (shouldn't happen, but be safe)
                # Include the message to avoid breaking the flow
                validated_messages.append(msg)
                i += 1
                continue
            
            # Check if ALL required tool_call_ids have ToolMessages in our list
            missing_tool_call_ids = required_tool_call_ids - set(all_tool_messages_by_id.keys())
            
            if missing_tool_call_ids:
                # Missing some tool responses - SKIP this AIMessage entirely
                # Also skip any ToolMessages that immediately follow (they belong to this incomplete sequence)
                i += 1
                # Skip ToolMessages that belong to this AIMessage
                while i < len(messages_to_add):
                    next_msg = messages_to_add[i]
                    if isinstance(next_msg, ToolMessage):
                        if (hasattr(next_msg, 'tool_call_id') and 
                            next_msg.tool_call_id and 
                            next_msg.tool_call_id in required_tool_call_ids):
                            # This ToolMessage belongs to the skipped AIMessage, skip it too
                            i += 1
                            continue
                    # Not a ToolMessage for this AIMessage, stop skipping
                    break
                continue
            
            # All tool_call_ids have responses - include the AIMessage
            validated_messages.append(msg)
            
            # Now include all ToolMessages that belong to this AIMessage
            # They should come immediately after the AIMessage
            j = i + 1
            included_tool_call_ids = set()
            while j < len(messages_to_add):
                next_msg = messages_to_add[j]
                
                # Stop if we hit another AIMessage or HumanMessage
                if isinstance(next_msg, (AIMessage, HumanMessage)):
                    break
                
                # Check if this is a ToolMessage for one of our tool_call_ids
                if isinstance(next_msg, ToolMessage):
                    tool_call_id = None
                    if hasattr(next_msg, 'tool_call_id'):
                        tool_call_id = next_msg.tool_call_id
                    elif isinstance(next_msg, dict):
                        tool_call_id = next_msg.get('tool_call_id')
                    
                    if tool_call_id and tool_call_id in required_tool_call_ids:
                        if tool_call_id not in included_tool_call_ids:
                            validated_messages.append(next_msg)
                            included_tool_call_ids.add(tool_call_id)
                        j += 1
                    else:
                        # This ToolMessage belongs to a different AIMessage, stop here
                        break
                else:
                    # Non-ToolMessage, stop here
                    break
            
            # Move past all processed messages
            i = j
            
        elif isinstance(msg, ToolMessage):
            # Orphaned ToolMessage - check if it was already included by an AIMessage
            # If not, it means its AIMessage was skipped, so we should skip it too
            tool_call_id = None
            if hasattr(msg, 'tool_call_id'):
                tool_call_id = msg.tool_call_id
            elif isinstance(msg, dict):
                tool_call_id = msg.get('tool_call_id')
            
            # Check if this ToolMessage's AIMessage is in our validated list
            # If the AIMessage is before this ToolMessage and has this tool_call_id, it was handled
            # Otherwise, skip this orphaned ToolMessage
            is_handled = False
            for validated_msg in validated_messages:
                if isinstance(validated_msg, AIMessage) and hasattr(validated_msg, 'tool_calls'):
                    for tool_call in validated_msg.tool_calls:
                        validated_tool_call_id = None
                        if hasattr(tool_call, 'id'):
                            validated_tool_call_id = tool_call.id
                        elif isinstance(tool_call, dict):
                            validated_tool_call_id = tool_call.get('id')
                        
                        if validated_tool_call_id == tool_call_id:
                            is_handled = True
                            break
                    if is_handled:
                        break
            
            if not is_handled:
                # This ToolMessage's AIMessage was not included, skip it
                i += 1
            else:
                # This should not happen - if it's handled, it should already be in validated_messages
                # But just in case, skip it to avoid duplicates
                i += 1
                
        else:
            # Regular message (HumanMessage, etc.), include it
            validated_messages.append(msg)
            i += 1
    
    result.extend(validated_messages)
    
    # Final safety check: Remove any AIMessage with tool_calls that doesn't have ALL ToolMessages
    # This is a last-ditch effort to prevent API errors
    final_cleaned_messages: List[BaseMessage] = []
    tool_messages_by_id: dict[str, ToolMessage] = {}
    
    # Preserve system message if present (it should be first)
    system_msg_in_result = None
    messages_to_process = result
    if result and isinstance(result[0], SystemMessage):
        system_msg_in_result = result[0]
        messages_to_process = result[1:]  # Process everything except system message
    
    # First pass: collect all ToolMessages
    for msg in messages_to_process:
        if isinstance(msg, ToolMessage):
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                tool_messages_by_id[msg.tool_call_id] = msg
    
    # Second pass: only include AIMessages with tool_calls if ALL tool_call_ids have ToolMessages
    i = 0
    while i < len(messages_to_process):
        msg = messages_to_process[i]
        
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            # Extract all tool_call_ids
            required_ids = set()
            for tool_call in msg.tool_calls:
                tool_call_id = None
                if hasattr(tool_call, 'id'):
                    tool_call_id = tool_call.id
                elif isinstance(tool_call, dict):
                    tool_call_id = tool_call.get('id')
                if tool_call_id:
                    required_ids.add(tool_call_id)
            
            # Check if ALL tool_call_ids have ToolMessages
            # Use strict equality check: all required_ids must be present
            all_tool_call_ids_present = bool(required_ids) and len(required_ids) > 0 and required_ids.issubset(set(tool_messages_by_id.keys()))
            
            if all_tool_call_ids_present:
                # Verify we can actually find all ToolMessages in the message list
                # Double-check by scanning ahead
                tool_messages_found = set()
                j = i + 1
                while j < len(messages_to_process):
                    next_msg = messages_to_process[j]
                    if isinstance(next_msg, (AIMessage, HumanMessage, SystemMessage)):
                        break
                    if isinstance(next_msg, ToolMessage):
                        tool_id = getattr(next_msg, 'tool_call_id', None)
                        if tool_id and tool_id in required_ids:
                            tool_messages_found.add(tool_id)
                        j += 1
                    else:
                        break
                
                # Only include if we found ALL tool_call_ids in the messages that follow
                if tool_messages_found == required_ids:
                    # All ToolMessages are present - include AIMessage
                    final_cleaned_messages.append(msg)
                    
                    # Include all ToolMessages that follow and belong to this AIMessage
                    j = i + 1
                    included_tool_ids = set()
                    while j < len(messages_to_process):
                        next_msg = messages_to_process[j]
                        if isinstance(next_msg, (AIMessage, HumanMessage, SystemMessage)):
                            break
                        if isinstance(next_msg, ToolMessage):
                            tool_id = getattr(next_msg, 'tool_call_id', None)
                            if tool_id and tool_id in required_ids and tool_id not in included_tool_ids:
                                final_cleaned_messages.append(next_msg)
                                included_tool_ids.add(tool_id)
                                j += 1
                            else:
                                break
                        else:
                            break
                    i = j
                else:
                    # Not all ToolMessages found in sequence - skip this AIMessage
                    i += 1
                    # Skip following ToolMessages that belong to this AIMessage
                    while i < len(messages_to_process):
                        next_msg = messages_to_process[i]
                        if isinstance(next_msg, ToolMessage):
                            tool_id = getattr(next_msg, 'tool_call_id', None)
                            if tool_id and tool_id in required_ids:
                                i += 1
                            else:
                                break
                        else:
                            break
            else:
                # Missing ToolMessages - skip this AIMessage and its ToolMessages
                i += 1
                # Skip following ToolMessages that belong to this AIMessage
                while i < len(messages_to_process):
                    next_msg = messages_to_process[i]
                    if isinstance(next_msg, ToolMessage):
                        tool_id = getattr(next_msg, 'tool_call_id', None) or (next_msg.get('tool_call_id') if isinstance(next_msg, dict) else None)
                        if tool_id and tool_id in required_ids:
                            i += 1
                        else:
                            break
                    else:
                        break
        elif isinstance(msg, ToolMessage):
            # Check if this ToolMessage's AIMessage is already in final_cleaned_messages
            tool_id = getattr(msg, 'tool_call_id', None) or (msg.get('tool_call_id') if isinstance(msg, dict) else None)
            is_included = False
            if tool_id:
                for final_msg in final_cleaned_messages:
                    if isinstance(final_msg, AIMessage) and hasattr(final_msg, 'tool_calls'):
                        for tc in final_msg.tool_calls:
                            tc_id = getattr(tc, 'id', None) or (tc.get('id') if isinstance(tc, dict) else None)
                            if tc_id == tool_id:
                                is_included = True
                                break
                        if is_included:
                            break
            
            if not is_included:
                # This ToolMessage's AIMessage was not included, skip it
                i += 1
            else:
                # Should already be included, skip to avoid duplicates
                i += 1
        else:
            # Regular message, include it
            final_cleaned_messages.append(msg)
            i += 1
    
    # Prepend system message if we have one
    if system_msg_in_result:
        return [system_msg_in_result] + final_cleaned_messages
    elif system_message:
        # Fallback: use the original system_message if system_msg_in_result is None
        return [system_message] + final_cleaned_messages
    
    return final_cleaned_messages

